"""Train/evaluate Step32 longer-horizon target diagnostics."""

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

from data.bridgedata_v2_tfds_context_selection import select_context_tokens, summarize_selection
from data.bridgedata_v2_tfds_context_utility_metrics import summarize_train_val_curve
from data.bridgedata_v2_tfds_longer_horizon_manifest import read_jsonl
from data.bridgedata_v2_tfds_longer_horizon_metrics import (
    summarize_longer_horizon_trainval,
    summarize_policy_val_losses,
    target_summary_for_variant,
)
from data.bridgedata_v2_tfds_world_model_batch import (
    load_bridge_tfds_world_model_samples,
    validate_world_model_sample,
)
from models.bridgedata_v2_context_bottleneck_smoke_model import BridgeDataContextBottleneckSmokePredictor
from scripts.train_eval_bridgedata_v2_tfds_context_utility import (
    OPTIMIZER_SCOPE,
    _optimizer_scope,
    _resolve_device,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_longer_horizon_step32.yaml"


def train_eval_step32_longer_horizon(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
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
        payload = _safe_stop_payload(config, f"missing Step32 trainval inputs: {missing}", env_guard)
        _write_outputs(config, payload)
        return payload

    loader_config = _loader_config(config)
    samples = load_bridge_tfds_world_model_samples(loader_config)
    window_meta = _window_metadata_by_sample(config["output"]["horizon_window_manifest_jsonl"])
    for sample in samples:
        validate_world_model_sample(sample)
        sample["horizon_gap"] = int(window_meta[str(sample["sample_id"])]["horizon_gap"])
        sample["metadata"]["horizon_gap"] = int(sample["horizon_gap"])
    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    splits_payload = json.loads(Path(config["output"]["horizon_split_json"]).read_text(encoding="utf-8"))

    train_cfg = config["tiny_trainval"]
    device = _resolve_device(str(train_cfg.get("device", "cpu")), bool(train_cfg.get("allow_cpu_fallback", True)))
    target_variants = [str(item) for item in config["targets"]["target_variants"]]
    split_runs: list[dict[str, Any]] = []
    optimizer_param_count = 0
    optimizer_step_performed = False

    for split_index, split in enumerate(splits_payload.get("splits", [])):
        horizon_gap = int(split["horizon_gap"])
        split_seed = int(split["split_seed"])
        train_samples = [samples_by_id[sample_id] for sample_id in split["train_sample_ids"] if sample_id in samples_by_id]
        val_samples = [samples_by_id[sample_id] for sample_id in split["val_sample_ids"] if sample_id in samples_by_id]
        if not train_samples or not val_samples:
            payload = _safe_stop_payload(config, f"horizon {horizon_gap} split {split_seed} has no train/val samples", env_guard)
            _write_outputs(config, payload)
            return payload
        for target_index, target_variant in enumerate(target_variants):
            _seed_everything(split_seed + horizon_gap * 17 + target_index * 100)
            policy_metrics: list[dict[str, Any]] = []
            loss_curves: list[dict[str, Any]] = []
            selection_summaries: list[dict[str, Any]] = []
            for policy_index, policy in enumerate(config["policies"]):
                if not bool(policy.get("train", True)):
                    continue
                policy_cfg = dict(policy)
                if str(policy_cfg.get("selection")) == "random":
                    policy_cfg["seed"] = int(policy_cfg.get("seed", 42)) + split_seed + horizon_gap
                model = BridgeDataContextBottleneckSmokePredictor(
                    hidden_dim=int(train_cfg.get("hidden_dim", 256)),
                    seed=int(config.get("seed", 42)) + split_seed + horizon_gap * 100 + target_index * 10 + policy_index,
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
                        "topk": train_data["topk"],
                        "horizon_gap": horizon_gap,
                        "target_variant": target_variant,
                        "split_seed": split_seed,
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
                        "horizon_gap": horizon_gap,
                        "target_variant": target_variant,
                        "split_seed": split_seed,
                        "policy": str(policy_cfg["name"]),
                        "topk": train_data["topk"],
                        "curve": curve,
                    }
                )

            policy_val = summarize_policy_val_losses(policy_metrics)
            row = {
                "horizon_gap": horizon_gap,
                "target_variant": target_variant,
                "split_seed": split_seed,
                "num_train_windows": len(train_samples),
                "num_val_windows": len(val_samples),
                "train_trajectories": split.get("train_trajectories", []),
                "val_trajectories": split.get("val_trajectories", []),
                "trajectory_disjoint": bool(split.get("train_val_trajectory_disjoint")),
                "fallback_used": bool(split.get("fallback_used")),
                "fallback_reason": split.get("fallback_reason"),
                "policy_metrics": policy_metrics,
                "policy_val": policy_val,
                "loss_curves": loss_curves,
                "selection_summaries": selection_summaries,
                "all_val_losses_finite": all(math.isfinite(float(value)) for value in policy_val.values()),
                "optimizer_step_scope": OPTIMIZER_SCOPE,
            }
            split_runs.append(row)

    positive = config["decision"].get("positive_signal_requires", {})
    horizon_summary = summarize_longer_horizon_trainval(
        split_runs,
        delta_gain_threshold=float(positive.get("delta_target_gain_over_mean_target_min", 0.10)),
    )
    decision = {
        "stage": config["stage"],
        "recommended_step33": horizon_summary["recommended_step33"],
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
        "tiny_trainval_diagnosis_performed": True,
        "tiny_trainval_training_performed": True,
        "training_performed": True,
        "num_selected_windows": len(samples),
        "num_runs": len(split_runs),
        "horizons_completed": sorted({int(sample["horizon_gap"]) for sample in samples}),
        "target_variants": target_variants,
        "split_seeds": [int(seed) for seed in config["window_selection"].get("split_seeds", [])],
        "policies_trained": [str(policy["name"]) for policy in config["policies"] if bool(policy.get("train", True))],
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
        "trainval_results": trainval,
        "horizon_target_summary": horizon_summary,
        "step32_decision": decision,
    }
    _write_outputs(config, payload)
    return payload


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
        selection = select_context_tokens(sample, policy)
        current_summary = sample["current_tokens"].detach().to(dtype=torch.float32).mean(dim=(0, 1))
        if selection["selected_context_tokens"].numel() == 0:
            context_summary = torch.zeros_like(current_summary)
        else:
            context_summary = selection["selected_context_tokens"].detach().to(dtype=torch.float32).mean(dim=0)
        current_summaries.append(current_summary)
        context_summaries.append(context_summary)
        targets.append(target_summary_for_variant(sample, target_variant))
        masses.append(float(selection["selected_importance_mass"]))
        selection_summaries.append(
            {"sample_id": str(sample["sample_id"]), "horizon_gap": int(sample["horizon_gap"]), **summarize_selection(selection)}
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


def _window_metadata_by_sample(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(record["sample_id"]): record for record in read_jsonl(path)}


def _missing_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["output"]["horizon_window_manifest_jsonl"],
        config["output"]["horizon_split_json"],
        config["output"]["token_manifest_jsonl"],
        config["output"]["token_summary_json"],
        config["output"]["token_smoke_dir"],
        config["output"]["importance_manifest_jsonl"],
        config["output"]["importance_summary_json"],
        config["output"]["importance_dir"],
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
    summary = {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "tiny_trainval_diagnosis_performed": False,
        "horizon_target_rows": [],
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "recommended_step33": {
            "name": "safe-stop: restore Step32 required inputs",
            "condition": reason,
            "scope": "no selector/current-importance training yet",
        },
        "safety_gate_pass": trainval["safety_gate_pass"],
    }
    decision = {
        "stage": config["stage"],
        "recommended_step33": summary["recommended_step33"],
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "safety_gate_pass": trainval["safety_gate_pass"],
    }
    return {"trainval_results": trainval, "horizon_target_summary": summary, "step32_decision": decision}


def _write_outputs(config: dict[str, Any], payload: dict[str, Any]) -> None:
    _write_json(Path(config["output"]["trainval_results_json"]), payload["trainval_results"])
    _write_json(Path(config["output"]["horizon_target_summary_json"]), payload["horizon_target_summary"])
    _write_json(Path(config["output"]["step32_decision_json"]), payload["step32_decision"])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _count_jsonl(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


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
    payload = train_eval_step32_longer_horizon(args.config)
    print(json.dumps(payload["horizon_target_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
