"""Evaluate Step30B teacher-topK tiny utility sanity."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_context_utility_metrics import summarize_train_val_curve
from data.bridgedata_v2_tfds_teacher_label_manifest import read_teacher_importance_manifest_jsonl, validate_teacher_importance_artifact
from data.bridgedata_v2_tfds_teacher_proxy_comparison import compare_teacher_proxy_importance
from data.bridgedata_v2_tfds_teacher_topk_metrics import (
    recommended_step31_from_teacher_topk,
    summarize_teacher_topk_utility,
)
from models.bridgedata_v2_context_bottleneck_smoke_model import BridgeDataContextBottleneckSmokePredictor
from scripts.train_eval_bridgedata_v2_tfds_context_utility import (
    OPTIMIZER_SCOPE,
    _optimizer_scope,
    _policy_dataset,
    _resolve_device,
    _train_policy,
)
from training.bridgedata_v2_tfds_occlusion_teacher_trainer import load_step30a_samples

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_occlusion_teacher_step30b.yaml"


def evaluate_step30b_teacher_topk(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    missing = _missing_inputs(config)
    if missing:
        payload = _safe_stop_payload(config, f"missing Step30B teacher-topK inputs: {missing}")
        _write_json(Path(config["output"]["teacher_topk_utility_json"]), payload["teacher_topk_utility"])
        _write_json(Path(config["output"]["teacher_decision_json"]), payload["teacher_decision"])
        return payload

    comparison = compare_teacher_proxy_importance(
        config["output"]["teacher_importance_manifest_jsonl"],
        config["input"]["proxy_importance_manifest_jsonl"],
        output_json=config["output"]["teacher_proxy_comparison_json"],
        topk_values=[int(k) for k in config["teacher_label_eval"]["compute_topk_overlap"]["topk_values"]],
    )
    samples = load_step30a_samples(config)
    teacher_by_id = _load_teacher_importance_by_id(config["output"]["teacher_importance_manifest_jsonl"])
    splits = json.loads(Path(config["input"]["multiseed_splits_json"]).read_text(encoding="utf-8"))
    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    utility_cfg = config["teacher_topk_utility"]
    split_seed_filter = {int(seed) for seed in utility_cfg.get("split_seeds", [42, 123, 999])}
    device = _resolve_device(str(config["teacher_training"].get("device", "cpu")), bool(config["teacher_training"].get("allow_cpu_fallback", True)))
    runs: list[dict[str, Any]] = []
    optimizer_step_performed = False
    optimizer_param_count = 0

    for split in splits.get("splits", []):
        split_seed = int(split["split_seed"])
        if split_seed not in split_seed_filter:
            continue
        train_samples = [samples_by_id[item] for item in split["train_sample_ids"] if item in samples_by_id]
        val_samples = [samples_by_id[item] for item in split["val_sample_ids"] if item in samples_by_id]
        policy_metrics: list[dict[str, Any]] = []
        for policy_index, policy in enumerate(_policies(int(utility_cfg.get("topk", 256)), split_seed)):
            policy_samples_train = _samples_for_policy(train_samples, policy, teacher_by_id)
            policy_samples_val = _samples_for_policy(val_samples, policy, teacher_by_id)
            seed = int(config.get("seed", 42)) + split_seed + policy_index
            _seed_everything(seed)
            model = BridgeDataContextBottleneckSmokePredictor(
                hidden_dim=int(utility_cfg.get("hidden_dim", 512)),
                seed=seed,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(utility_cfg.get("learning_rate", 0.001)),
                weight_decay=0.0,
            )
            scope = _optimizer_scope(model, optimizer)
            if scope["optimizer_step_scope"] != OPTIMIZER_SCOPE:
                raise RuntimeError(f"optimizer scope must be {OPTIMIZER_SCOPE}: {scope}")
            optimizer_param_count = max(optimizer_param_count, int(scope["optimizer_param_count"]))
            train_data = _policy_dataset(policy_samples_train, policy, device)
            val_data = _policy_dataset(policy_samples_val, policy, device)
            curve, performed = _train_policy(
                model=model,
                optimizer=optimizer,
                train_data=train_data,
                val_data=val_data,
                train_steps=int(utility_cfg.get("train_steps", 500)),
                eval_every=int(utility_cfg.get("eval_every", 25)),
                grad_clip_norm=float(config["teacher_training"].get("grad_clip_norm", 1.0)),
            )
            optimizer_step_performed = optimizer_step_performed or performed
            metric = summarize_train_val_curve(str(policy["name"]), curve)
            metric.update(
                {
                    "split_seed": split_seed,
                    "topk": train_data["topk"],
                    "num_train_windows": len(policy_samples_train),
                    "num_val_windows": len(policy_samples_val),
                    "mean_selected_importance_mass_train": train_data["mean_selected_importance_mass"],
                    "mean_selected_importance_mass_val": val_data["mean_selected_importance_mass"],
                    "optimizer_step_performed": bool(performed),
                    "optimizer_step_scope": OPTIMIZER_SCOPE,
                    "current_tokens_kept_full": True,
                    "train_current_importance": False,
                }
            )
            policy_metrics.append(metric)
        runs.append(
            {
                "split_seed": split_seed,
                "num_train_windows": len(train_samples),
                "num_val_windows": len(val_samples),
                "trajectory_disjoint": bool(split.get("train_val_trajectory_disjoint")),
                "policy_metrics": policy_metrics,
                "optimizer_step_scope": OPTIMIZER_SCOPE,
            }
        )

    summary = summarize_teacher_topk_utility(runs)
    summary.update(
        {
            "optimizer_step_performed": bool(optimizer_step_performed),
            "optimizer_step_scope": OPTIMIZER_SCOPE,
            "optimizer_param_count": int(optimizer_param_count),
            "runs": runs,
            "trained_predictor_teacher_training_performed": True,
            "teacher_training_scope": "small_predictor_teacher_only",
            "selector_training_performed": False,
            "current_importance_training_performed": False,
            "checkpoint_saved": False,
        }
    )
    decision = {
        "stage": config["stage"],
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "do_not_train_selector_yet": True,
        "do_not_train_current_importance_yet": True,
        "never_claim_final_utility": True,
        "recommended_step31": recommended_step31_from_teacher_topk(
            summary,
            teacher_label_nontrivial=bool(comparison.get("teacher_label_nontrivial")),
        ),
        "teacher_topk_signal": summary["teacher_topk_signal"],
        "safety_gate_pass": True,
    }
    _write_json(Path(config["output"]["teacher_topk_utility_json"]), summary)
    _write_json(Path(config["output"]["teacher_decision_json"]), decision)
    return {
        "teacher_proxy_comparison": comparison,
        "teacher_topk_utility": summary,
        "teacher_decision": decision,
    }


def _policies(topk: int, split_seed: int) -> list[dict[str, Any]]:
    return [
        {"name": "current_only", "use_context": False, "topk": 0, "train": True},
        {"name": "random_context_topk", "use_context": True, "selection": "random", "topk": topk, "seed": split_seed, "train": True},
        {"name": "proxy_importance_topk", "use_context": True, "selection": "importance_topk", "topk": topk, "train": True},
        {"name": "teacher_occlusion_topk", "use_context": True, "selection": "importance_topk", "topk": topk, "train": True},
        {"name": "full_context_reference", "use_context": True, "selection": "full", "topk": None, "deployable": False, "train": True},
    ]


def _samples_for_policy(samples: list[dict[str, Any]], policy: dict[str, Any], teacher_by_id: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    if str(policy["name"]) != "teacher_occlusion_topk":
        return samples
    output = []
    for sample in samples:
        copied = dict(sample)
        copied["context_importance"] = teacher_by_id[str(sample["sample_id"])].detach().cpu().to(dtype=torch.float32)
        output.append(copied)
    return output


def _load_teacher_importance_by_id(manifest_path: str | Path) -> dict[str, torch.Tensor]:
    records = read_teacher_importance_manifest_jsonl(manifest_path)
    result: dict[str, torch.Tensor] = {}
    for record in records:
        artifact = validate_teacher_importance_artifact(record["importance_artifact_path"])
        result[str(record["sample_id"])] = artifact["context_importance_norm"].detach().cpu().to(dtype=torch.float32)
    return result


def _missing_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["output"]["teacher_importance_manifest_jsonl"],
        config["output"]["teacher_importance_summary_json"],
        config["input"]["proxy_importance_manifest_jsonl"],
        config["input"]["multiseed_splits_json"],
        config["input"]["token_manifest_jsonl"],
        config["input"]["token_summary_json"],
    ]
    return [str(path) for path in required if not Path(path).exists()]


def _safe_stop_payload(config: dict[str, Any], reason: str) -> dict[str, Any]:
    summary = {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "teacher_topk_utility_performed": False,
        "runs": [],
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "safety_gate_pass": True,
    }
    decision = {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "safety_gate_pass": True,
    }
    return {"teacher_topk_utility": summary, "teacher_decision": decision}


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    payload = evaluate_step30b_teacher_topk(args.config)
    print(json.dumps(payload["teacher_topk_utility"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
