"""Train/evaluate Step33A true-temporal representation diagnostics."""

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
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_context_utility_metrics import summarize_train_val_curve
from data.bridgedata_v2_tfds_true_temporal_manifest import load_step32_gap0_splits
from data.bridgedata_v2_tfds_true_temporal_metrics import (
    summarize_policy_val_losses,
    summarize_true_temporal_trainval,
)
from data.bridgedata_v2_tfds_true_temporal_proxy_importance import (
    load_true_temporal_importance_artifact,
    read_true_temporal_importance_manifest,
)
from data.bridgedata_v2_tfds_true_temporal_schema import (
    load_true_temporal_token_artifact,
    read_true_temporal_manifest,
    select_true_temporal_context_tokens,
    target_summary_for_variant,
)
from models.bridgedata_v2_context_bottleneck_smoke_model import BridgeDataContextBottleneckSmokePredictor
from scripts.train_eval_bridgedata_v2_tfds_context_utility import (
    OPTIMIZER_SCOPE,
    _optimizer_scope,
    _resolve_device,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_true_temporal_step33a.yaml"


def train_eval_step33a_true_temporal(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
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
        payload = _safe_stop_payload(config, f"missing Step33A train/eval inputs: {missing}", env_guard)
        _write_outputs(config, payload)
        return payload

    samples = _load_true_temporal_samples(config)
    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    splits = load_step32_gap0_splits(config["input"]["step32_horizon_splits_json"], set(samples_by_id))
    if not splits:
        payload = _safe_stop_payload(config, "no reusable Step32 gap0 train/val splits found", env_guard)
        _write_outputs(config, payload)
        return payload

    train_cfg = config["tiny_trainval"]
    device = _resolve_device(str(train_cfg.get("device", "cpu")), bool(train_cfg.get("allow_cpu_fallback", True)))
    target_variants = [str(item) for item in config["comparison"]["target_variants"]]
    policy_cfgs = [dict(policy) for policy in config["policies"] if bool(policy.get("train", True))]
    split_runs: list[dict[str, Any]] = []
    optimizer_param_count = 0
    optimizer_step_performed = False

    for split in splits:
        split_seed = int(split["split_seed"])
        train_samples = [samples_by_id[sample_id] for sample_id in split["train_sample_ids"] if sample_id in samples_by_id]
        val_samples = [samples_by_id[sample_id] for sample_id in split["val_sample_ids"] if sample_id in samples_by_id]
        if not train_samples or not val_samples:
            payload = _safe_stop_payload(config, f"gap0 split {split_seed} has no train/val samples", env_guard)
            _write_outputs(config, payload)
            return payload
        for target_index, target_variant in enumerate(target_variants):
            _seed_everything(split_seed + target_index * 100)
            policy_metrics: list[dict[str, Any]] = []
            loss_curves: list[dict[str, Any]] = []
            selection_summaries: list[dict[str, Any]] = []
            for policy_index, policy in enumerate(policy_cfgs):
                policy_cfg = dict(policy)
                if str(policy_cfg.get("selection")) == "random":
                    policy_cfg["seed"] = int(policy_cfg.get("seed", 42)) + split_seed
                model = BridgeDataContextBottleneckSmokePredictor(
                    hidden_dim=int(train_cfg.get("hidden_dim", 256)),
                    seed=int(config.get("seed", 42)) + split_seed + target_index * 10 + policy_index,
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
                train_data = _policy_dataset(train_samples, policy_cfg, target_variant, device)
                val_data = _policy_dataset(val_samples, policy_cfg, target_variant, device)
                selection_summaries.extend(train_data["selection_summaries"])
                selection_summaries.extend(val_data["selection_summaries"])
                curve, performed = _train_policy(
                    model=model,
                    optimizer=optimizer,
                    train_data=train_data,
                    val_data=val_data,
                    train_steps=int(train_cfg.get("train_steps", 400)),
                    eval_every=int(train_cfg.get("eval_every", 50)),
                    grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
                )
                optimizer_step_performed = optimizer_step_performed or performed
                metrics = summarize_train_val_curve(str(policy_cfg["name"]), curve)
                metrics.update(
                    {
                        "representation_mode": "true_temporal_clip_step33a",
                        "horizon_gap": 0,
                        "target_variant": target_variant,
                        "split_seed": split_seed,
                        "topk": train_data["topk"],
                        "num_train_windows": len(train_samples),
                        "num_val_windows": len(val_samples),
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
                    {
                        "representation_mode": "true_temporal_clip_step33a",
                        "horizon_gap": 0,
                        "target_variant": target_variant,
                        "split_seed": split_seed,
                        "policy": str(policy_cfg["name"]),
                        "topk": train_data["topk"],
                        "curve": curve,
                    }
                )
            policy_val = summarize_policy_val_losses(policy_metrics)
            split_runs.append(
                {
                    "representation_mode": "true_temporal_clip_step33a",
                    "horizon_gap": 0,
                    "target_variant": target_variant,
                    "split_seed": split_seed,
                    "num_train_windows": len(train_samples),
                    "num_val_windows": len(val_samples),
                    "train_trajectories": split.get("train_trajectories", []),
                    "val_trajectories": split.get("val_trajectories", []),
                    "trajectory_disjoint": bool(split.get("train_val_trajectory_disjoint")),
                    "policy_metrics": policy_metrics,
                    "policy_val": policy_val,
                    "loss_curves": loss_curves,
                    "selection_summaries": selection_summaries,
                    "all_val_losses_finite": all(math.isfinite(float(value)) for value in policy_val.values()),
                    "optimizer_step_scope": OPTIMIZER_SCOPE,
                }
            )

    step32_summary = _read_json(config["input"]["step32_horizon_target_summary_json"])
    comparison = summarize_true_temporal_trainval(
        split_runs,
        step32_horizon_target_summary=step32_summary,
        positive_thresholds=config["decision"].get("true_temporal_positive_requires", {}),
    )
    decision = {
        "stage": config["stage"],
        "horizon_gap": 0,
        "target_variant": config["comparison"]["primary_target"],
        "recommended_step33b_or_step34": comparison["recommended_step33b_or_step34"],
        "true_temporal_representation_helped": bool(comparison["true_temporal_representation_helped"]),
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "do_not_train_selector_yet": True,
        "do_not_train_current_importance_yet": True,
        "never_claim_final_utility": True,
        "safety_gate_pass": True,
    }
    trainval = {
        "stage": config["stage"],
        "safe_stop": False,
        "reason": None,
        "representation_mode": "true_temporal_clip_step33a",
        "tiny_trainval_diagnosis_performed": True,
        "tiny_trainval_training_performed": True,
        "training_performed": True,
        "num_selected_windows": len(samples),
        "num_runs": len(split_runs),
        "horizon_gap": 0,
        "target_variants": target_variants,
        "split_seeds": [int(split["split_seed"]) for split in splits],
        "policies_trained": [str(policy["name"]) for policy in policy_cfgs],
        "train_steps": int(train_cfg.get("train_steps", 400)),
        "optimizer": str(train_cfg.get("optimizer", "adamw")),
        "optimizer_step_performed": bool(optimizer_step_performed),
        "optimizer_step_scope": OPTIMIZER_SCOPE,
        "optimizer_param_count": int(optimizer_param_count),
        "all_val_losses_finite": all(bool(run["all_val_losses_finite"]) for run in split_runs),
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
        "limited_true_temporal_token_extraction_performed": True,
        "limited_true_temporal_proxy_importance_performed": True,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "checkpoint_saved": False,
        "tiny_trainval_result_not_final_performance": True,
        "runs": split_runs,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": True,
    }
    payload = {
        "true_temporal_trainval_results": trainval,
        "representation_comparison": comparison,
        "step33a_decision": decision,
    }
    _write_outputs(config, payload)
    return payload


def _load_true_temporal_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    token_records = read_true_temporal_manifest(config["output"]["true_temporal_token_manifest_jsonl"])
    importance_records = read_true_temporal_importance_manifest(
        config["output"]["true_temporal_importance_manifest_jsonl"]
    )
    importance_by_id = {str(record["sample_id"]): record for record in importance_records}
    samples: list[dict[str, Any]] = []
    for record in token_records:
        sample_id = str(record["sample_id"])
        if sample_id not in importance_by_id:
            raise FileNotFoundError(f"missing Step33A importance record for sample_id={sample_id}")
        sample = load_true_temporal_token_artifact(record)
        importance_artifact = load_true_temporal_importance_artifact(importance_by_id[sample_id])
        sample["context_importance"] = importance_artifact["context_importance_norm"].detach().to(
            dtype=torch.float32, device="cpu"
        )
        sample["metadata"]["importance_artifact_path"] = importance_by_id[sample_id]["importance_artifact_path"]
        samples.append(sample)
    return samples


def _policy_dataset(
    samples: list[dict[str, Any]],
    policy: dict[str, Any],
    target_variant: str,
    device: torch.device,
) -> dict[str, Any]:
    current_summaries = []
    context_summaries = []
    targets = []
    selection_summaries = []
    masses = []
    for sample in samples:
        selection = select_true_temporal_context_tokens(sample, policy)
        current_summary = sample["current_summary"].detach().to(dtype=torch.float32)
        if selection["selected_context_tokens"].numel() == 0:
            context_summary = torch.zeros_like(current_summary)
        else:
            context_summary = selection["selected_context_tokens"].detach().to(dtype=torch.float32).mean(dim=0)
        current_summaries.append(current_summary)
        context_summaries.append(context_summary)
        targets.append(target_summary_for_variant(sample, target_variant))
        masses.append(float(selection["selected_importance_mass"]))
        selection_summaries.append(
            {
                "sample_id": str(sample["sample_id"]),
                "horizon_gap": 0,
                "policy_name": selection["policy_name"],
                "num_selected": int(selection["num_selected"]),
                "topk": selection["topk"],
                "selected_context_shape": list(selection["selected_context_tokens"].shape),
                "selected_importance_mass": float(selection["selected_importance_mass"]),
                "selected_importance_mean": float(selection["selected_importance_mean"]),
                "current_tokens_kept_full": True,
                "deployable": bool(selection.get("deployable", True)),
            }
        )
    return {
        "current_summaries": torch.stack(current_summaries, dim=0).to(device),
        "context_summaries": torch.stack(context_summaries, dim=0).to(device),
        "targets": torch.stack(targets, dim=0).to(device),
        "selection_summaries": selection_summaries,
        "mean_selected_importance_mass": float(sum(masses) / len(masses)) if masses else 0.0,
        "topk": selection_summaries[0]["topk"] if selection_summaries else policy.get("topk"),
    }


def _train_policy(
    model: BridgeDataContextBottleneckSmokePredictor,
    optimizer: torch.optim.Optimizer,
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    train_steps: int,
    eval_every: int,
    grad_clip_norm: float,
) -> tuple[list[dict[str, Any]], bool]:
    curve = [_curve_point(0, model, train_data, val_data)]
    performed = False
    for step in range(1, train_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            pred = model.forward_from_summaries(train_data["current_summaries"], train_data["context_summaries"])
            loss = F.mse_loss(pred, train_data["targets"], reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        performed = True
        if step == 1 or step % max(eval_every, 1) == 0 or step == train_steps:
            curve.append(_curve_point(step, model, train_data, val_data))
    return curve, performed


def _curve_point(
    step: int,
    model: BridgeDataContextBottleneckSmokePredictor,
    train_data: dict[str, Any],
    val_data: dict[str, Any],
) -> dict[str, Any]:
    return {"step": int(step), "train_loss": _evaluate_loss(model, train_data), "val_loss": _evaluate_loss(model, val_data)}


def _evaluate_loss(model: BridgeDataContextBottleneckSmokePredictor, data: dict[str, Any]) -> float:
    model.eval()
    with torch.no_grad():
        pred = model.forward_from_summaries(data["current_summaries"], data["context_summaries"])
        loss = F.mse_loss(pred, data["targets"], reduction="mean")
    return float(loss.item())


def _missing_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["input"]["step32_horizon_splits_json"],
        config["input"]["step32_horizon_target_summary_json"],
        config["output"]["true_temporal_token_manifest_jsonl"],
        config["output"]["true_temporal_token_summary_json"],
        config["output"]["true_temporal_importance_manifest_jsonl"],
        config["output"]["true_temporal_importance_summary_json"],
    ]
    return [str(path) for path in required if not Path(path).exists()]


def _safe_stop_payload(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    trainval = {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "tiny_trainval_diagnosis_performed": False,
        "tiny_trainval_training_performed": False,
        "training_performed": False,
        "num_selected_windows": 0,
        "num_runs": 0,
        "optimizer_step_performed": False,
        "optimizer_step_scope": OPTIMIZER_SCOPE,
        "all_val_losses_finite": False,
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
    comparison = {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "tiny_trainval_diagnosis_performed": False,
        "frame_repeat_proxy_gain_mean": 0.0,
        "true_temporal_proxy_gain_mean": 0.0,
        "true_temporal_gain_over_frame_repeat": 0.0,
        "true_temporal_proxy_beats_current_fraction": 0.0,
        "true_temporal_proxy_beats_random_fraction": 0.0,
        "true_temporal_full_context_noise_confirmed": False,
        "true_temporal_representation_helped": False,
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "recommended_step33b_or_step34": {
            "name": "safe-stop: restore Step33A required inputs",
            "condition": reason,
            "scope": "no selector/current-importance training yet",
        },
        "safety_gate_pass": trainval["safety_gate_pass"],
    }
    decision = {
        "stage": config["stage"],
        "recommended_step33b_or_step34": comparison["recommended_step33b_or_step34"],
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "safety_gate_pass": trainval["safety_gate_pass"],
    }
    return {
        "true_temporal_trainval_results": trainval,
        "representation_comparison": comparison,
        "step33a_decision": decision,
    }


def _write_outputs(config: dict[str, Any], payload: dict[str, Any]) -> None:
    _write_json(Path(config["output"]["true_temporal_trainval_results_json"]), payload["true_temporal_trainval_results"])
    _write_json(Path(config["output"]["representation_comparison_json"]), payload["representation_comparison"])
    _write_json(Path(config["output"]["step33a_decision_json"]), payload["step33a_decision"])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


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
    payload = train_eval_step33a_true_temporal(args.config)
    print(json.dumps(payload["representation_comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
