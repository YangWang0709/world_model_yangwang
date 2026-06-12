"""Train/evaluate Step30A multi-seed tiny context utility sanity models."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_context_signal_stability import (
    analyze_context_signal_stability,
    build_context_signal_decision,
)
from data.bridgedata_v2_tfds_context_utility_metrics import (
    build_context_utility_comparison,
    summarize_train_val_curve,
)
from data.bridgedata_v2_tfds_world_model_batch import (
    load_bridge_tfds_world_model_samples,
    validate_world_model_sample,
)
from models.bridgedata_v2_context_bottleneck_smoke_model import BridgeDataContextBottleneckSmokePredictor
from scripts.train_eval_bridgedata_v2_tfds_context_utility import (
    OPTIMIZER_SCOPE,
    _optimizer_scope,
    _policy_dataset,
    _resolve_device,
    _train_policy,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_window_diversity_step30a.yaml"


def train_eval_step30a_window_diversity(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(config_path)
    env_guard = _env_guard()
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        payload = _safe_stop_payload(config, "env_isaaclab is polluted with TensorFlow/TFDS", env_guard)
        _write_outputs(config, payload)
        return payload
    missing = _missing_inputs(config)
    if missing:
        payload = _safe_stop_payload(config, f"missing Step30A trainval inputs: {missing}", env_guard)
        _write_outputs(config, payload)
        return payload

    loader_config = _loader_config(config)
    samples = load_bridge_tfds_world_model_samples(loader_config)
    for sample in samples:
        validate_world_model_sample(sample)
    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    splits_payload = json.loads(Path(config["output"]["multiseed_splits_json"]).read_text(encoding="utf-8"))
    split_runs: list[dict[str, Any]] = []
    train_cfg = config["tiny_trainval"]
    positive_requires = config.get("stability_decision", {}).get("positive_requires", {})
    device = _resolve_device(str(train_cfg.get("device", "cpu")), bool(train_cfg.get("allow_cpu_fallback", True)))
    optimizer_param_count = 0
    optimizer_step_performed = False

    for split_index, split in enumerate(splits_payload.get("splits", [])):
        split_seed = int(split["split_seed"])
        _seed_everything(split_seed)
        train_samples = [samples_by_id[sample_id] for sample_id in split["train_sample_ids"] if sample_id in samples_by_id]
        val_samples = [samples_by_id[sample_id] for sample_id in split["val_sample_ids"] if sample_id in samples_by_id]
        if not train_samples or not val_samples:
            payload = _safe_stop_payload(config, f"split {split_seed} has no loaded train or val samples", env_guard)
            _write_outputs(config, payload)
            return payload

        policy_metrics: list[dict[str, Any]] = []
        loss_curves: list[dict[str, Any]] = []
        selection_summaries: list[dict[str, Any]] = []
        for policy_index, policy in enumerate(config["policies"]):
            if not bool(policy.get("train", True)):
                continue
            policy_cfg = dict(policy)
            if str(policy_cfg.get("selection")) == "random":
                policy_cfg["seed"] = int(policy_cfg.get("seed", 42)) + split_seed
            model = BridgeDataContextBottleneckSmokePredictor(
                hidden_dim=int(train_cfg.get("hidden_dim", 512)),
                seed=int(config.get("seed", 42)) + split_seed + policy_index + split_index * 1000,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(train_cfg.get("learning_rate", 0.001)),
                weight_decay=float(train_cfg.get("weight_decay", 0.0)),
            )
            scope = _optimizer_scope(model, optimizer)
            if scope["optimizer_step_scope"] != OPTIMIZER_SCOPE:
                raise RuntimeError(f"optimizer scope must be {OPTIMIZER_SCOPE}: {scope}")
            optimizer_param_count = max(optimizer_param_count, int(scope["optimizer_param_count"]))
            train_data = _policy_dataset(train_samples, policy_cfg, device)
            val_data = _policy_dataset(val_samples, policy_cfg, device)
            selection_summaries.extend(train_data["selection_summaries"])
            selection_summaries.extend(val_data["selection_summaries"])
            curve, performed = _train_policy(
                model=model,
                optimizer=optimizer,
                train_data=train_data,
                val_data=val_data,
                train_steps=int(train_cfg.get("train_steps", 500)),
                eval_every=int(train_cfg.get("eval_every", 25)),
                grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
            )
            optimizer_step_performed = optimizer_step_performed or performed
            metrics = summarize_train_val_curve(str(policy_cfg["name"]), curve)
            metrics.update(
                {
                    "topk": train_data["topk"],
                    "num_train_windows": len(train_samples),
                    "num_val_windows": len(val_samples),
                    "split_seed": split_seed,
                    "mean_selected_importance_mass_train": train_data["mean_selected_importance_mass"],
                    "mean_selected_importance_mass_val": val_data["mean_selected_importance_mass"],
                    "optimizer_step_performed": bool(performed),
                    "optimizer_step_scope": OPTIMIZER_SCOPE,
                    "current_tokens_kept_full": True,
                    "train_current_importance": False,
                }
            )
            policy_metrics.append(metrics)
            loss_curves.append(
                {"split_seed": split_seed, "policy": str(policy_cfg["name"]), "topk": train_data["topk"], "curve": curve}
            )

        comparison = build_context_utility_comparison(policy_metrics, positive_requires=positive_requires)
        split_runs.append(
            {
                "split_seed": split_seed,
                "num_train_windows": len(train_samples),
                "num_val_windows": len(val_samples),
                "train_trajectories": split.get("train_trajectories", []),
                "val_trajectories": split.get("val_trajectories", []),
                "trajectory_disjoint": bool(split.get("train_val_trajectory_disjoint")),
                "fallback_used": bool(split.get("fallback_used")),
                "fallback_reason": split.get("fallback_reason"),
                "policy_metrics": policy_metrics,
                "context_comparison": comparison,
                "loss_curves": loss_curves,
                "selection_summaries": selection_summaries,
                "all_val_losses_finite": bool(comparison.get("all_val_losses_finite")),
                "optimizer_step_scope": OPTIMIZER_SCOPE,
            }
        )

    stability_summary = analyze_context_signal_stability(
        {"runs": split_runs},
        positive_requires=positive_requires,
    )
    decision = build_context_signal_decision(stability_summary)
    trainval_runs = {
        "stage": config["stage"],
        "safe_stop": False,
        "reason": None,
        "tiny_trainval_training_performed": True,
        "training_performed": True,
        "num_selected_windows": len(samples),
        "num_splits": len(split_runs),
        "split_seeds": [int(run["split_seed"]) for run in split_runs],
        "policies_trained": [str(policy["name"]) for policy in config["policies"] if bool(policy.get("train", True))],
        "train_steps": int(train_cfg.get("train_steps", 500)),
        "optimizer": str(train_cfg.get("optimizer", "adamw")),
        "optimizer_step_performed": bool(optimizer_step_performed),
        "optimizer_step_scope": OPTIMIZER_SCOPE,
        "optimizer_param_count": int(optimizer_param_count),
        "all_val_losses_finite": all(bool(run["all_val_losses_finite"]) for run in split_runs),
        "context_signal_stability": stability_summary["context_signal_stability"],
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "current_importance_training_performed": False,
        "videomae_training_performed": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "world_model_large_training_performed": False,
        "action_conditioned_world_model_training_performed": False,
        "new_tfds_shard_downloaded": False,
        "download_performed": False,
        "raw_zip_downloaded": False,
        "full_tfds_downloaded": False,
        "droid_downloaded": False,
        "model_download_performed": False,
        "limited_token_extraction_performed": True,
        "limited_proxy_importance_generation_performed": True,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "checkpoint_saved": False,
        "tiny_trainval_result_not_final_performance": True,
        "runs": split_runs,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": True,
    }
    payload = {
        "trainval_runs": trainval_runs,
        "stability_summary": stability_summary,
        "context_signal_decision": decision,
    }
    _write_outputs(config, payload)
    return payload


def _loader_config(config: dict[str, Any]) -> dict[str, Any]:
    loader_config = dict(config)
    loader_config["input"] = dict(config["input"])
    loader_config["input"].update(
        {
            "token_manifest_jsonl": config["output"]["token_manifest_jsonl"],
            "token_summary_json": config["output"]["token_summary_json"],
            "token_smoke_dir": config["output"]["token_smoke_dir"],
            "importance_manifest_jsonl": config["output"]["importance_manifest_jsonl"],
            "importance_summary_json": config["output"]["importance_summary_json"],
            "importance_smoke_dir": config["output"]["importance_dir"],
        }
    )
    loader_config["sample_limits"] = dict(config.get("sample_limits", {}))
    loader_config["sample_limits"]["max_samples"] = _count_jsonl(config["output"]["token_manifest_jsonl"])
    return loader_config


def _missing_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["output"]["multiseed_splits_json"],
        config["output"]["token_manifest_jsonl"],
        config["output"]["token_summary_json"],
        config["output"]["token_smoke_dir"],
        config["output"]["importance_manifest_jsonl"],
        config["output"]["importance_summary_json"],
        config["output"]["importance_dir"],
    ]
    return [str(path) for path in required if not Path(path).exists()]


def _safe_stop_payload(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    trainval_runs = {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "tiny_trainval_training_performed": False,
        "training_performed": False,
        "num_selected_windows": 0,
        "num_splits": 0,
        "split_seeds": [],
        "policies_trained": [],
        "optimizer_step_performed": False,
        "optimizer_step_scope": OPTIMIZER_SCOPE,
        "all_val_losses_finite": False,
        "context_signal_stability": "safe_stop",
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "current_importance_training_performed": False,
        "videomae_training_performed": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "new_tfds_shard_downloaded": False,
        "model_download_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "checkpoint_saved": False,
        "runs": [],
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }
    stability_summary = {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "num_seeds": 0,
        "context_signal_stability": "safe_stop",
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "safety_gate_pass": trainval_runs["safety_gate_pass"],
    }
    decision = {
        "stage": config["stage"],
        "context_signal_stability": "safe_stop",
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "reason": reason,
        "safety_gate_pass": trainval_runs["safety_gate_pass"],
    }
    return {
        "trainval_runs": trainval_runs,
        "stability_summary": stability_summary,
        "context_signal_decision": decision,
    }


def _write_outputs(config: dict[str, Any], payload: dict[str, Any]) -> None:
    _write_json(Path(config["output"]["trainval_runs_json"]), payload["trainval_runs"])
    _write_json(Path(config["output"]["stability_summary_json"]), payload["stability_summary"])
    _write_json(Path(config["output"]["context_signal_decision_json"]), payload["context_signal_decision"])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _count_jsonl(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _finite_metric(item: dict[str, Any], key: str) -> bool:
    return math.isfinite(float(item.get(key, math.nan)))


def _env_guard() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    payload = train_eval_step30a_window_diversity(args.config)
    print(json.dumps(payload["trainval_runs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
