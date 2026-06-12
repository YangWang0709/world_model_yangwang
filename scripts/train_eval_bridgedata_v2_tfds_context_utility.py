"""Train/evaluate Step29 tiny train-val context utility sanity models."""

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
from data.bridgedata_v2_tfds_context_utility_metrics import (
    build_context_utility_comparison,
    summarize_train_val_curve,
)
from data.bridgedata_v2_tfds_world_model_batch import (
    load_bridge_tfds_world_model_samples,
    validate_world_model_sample,
)
from models.bridgedata_v2_context_bottleneck_smoke_model import BridgeDataContextBottleneckSmokePredictor

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_trainval_context_utility_step29.yaml"
OPTIMIZER_SCOPE = "tiny_world_model_predictor_only"


def train_eval_step29_context_utility(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(config_path)
    env_guard = _env_guard()
    paths = _output_paths(config)
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        payload = _safe_stop_payload(config, "env_isaaclab is polluted with TensorFlow/TFDS", env_guard)
        _write_outputs(paths, payload)
        return payload
    missing = _missing_inputs(config)
    if missing:
        payload = _safe_stop_payload(config, f"missing Step29 trainval inputs: {missing}", env_guard)
        _write_outputs(paths, payload)
        return payload

    samples = load_bridge_tfds_world_model_samples(config)
    for sample in samples:
        validate_world_model_sample(sample)
    split = json.loads(Path(config["output"]["split_json"]).read_text(encoding="utf-8"))
    train_ids = set(split["train_sample_ids"])
    val_ids = set(split["val_sample_ids"])
    train_samples = [sample for sample in samples if str(sample["sample_id"]) in train_ids]
    val_samples = [sample for sample in samples if str(sample["sample_id"]) in val_ids]
    if not train_samples or not val_samples:
        payload = _safe_stop_payload(config, "train or val split has no loaded samples", env_guard)
        _write_outputs(paths, payload)
        return payload

    seed = int(config.get("seed", 42))
    _seed_everything(seed)
    train_cfg = config["tiny_trainval"]
    acceptance = config.get("context_utility_acceptance", {})
    device = _resolve_device(str(train_cfg.get("device", "cpu")), bool(train_cfg.get("allow_cpu_fallback", True)))
    policy_metrics: list[dict[str, Any]] = []
    loss_curves: list[dict[str, Any]] = []
    selection_summaries: list[dict[str, Any]] = []
    optimizer_step_performed = False
    optimizer_param_count = 0

    for policy_index, policy in enumerate(config["policies"]):
        if not bool(policy.get("train", True)):
            continue
        model = BridgeDataContextBottleneckSmokePredictor(
            hidden_dim=int(train_cfg.get("hidden_dim", 512)),
            seed=seed + policy_index,
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
        train_data = _policy_dataset(train_samples, policy, device)
        val_data = _policy_dataset(val_samples, policy, device)
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
        metrics = summarize_train_val_curve(
            str(policy["name"]),
            curve,
            min_relative_train_loss_decrease=float(acceptance.get("min_relative_train_loss_decrease", 0.01)),
        )
        metrics.update(
            {
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
        loss_curves.append({"policy": str(policy["name"]), "topk": train_data["topk"], "curve": curve})

    comparison = build_context_utility_comparison(
        policy_metrics,
        positive_requires=acceptance.get("context_utility_positive_requires", {}),
    )
    summary = {
        "stage": config["stage"],
        "safe_stop": False,
        "reason": None,
        "tiny_trainval_training_performed": True,
        "training_performed": True,
        "num_selected_windows": len(samples),
        "num_train_windows": len(train_samples),
        "num_val_windows": len(val_samples),
        "policies_trained": [item["policy"] for item in policy_metrics],
        "train_steps": int(train_cfg.get("train_steps", 500)),
        "optimizer": str(train_cfg.get("optimizer", "adamw")),
        "optimizer_step_performed": bool(optimizer_step_performed),
        "optimizer_step_scope": OPTIMIZER_SCOPE,
        "optimizer_param_count": int(optimizer_param_count),
        "all_train_losses_finite": all(_finite_metric(item, "train_final_loss") for item in policy_metrics),
        "all_val_losses_finite": bool(comparison.get("all_val_losses_finite")),
        "context_utility_sanity_signal": comparison["context_utility_sanity_signal"],
        "context_utility_claim_allowed": False,
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
        "train_val_trajectory_disjoint": bool(split.get("train_val_trajectory_disjoint")),
        "policy_metric_table": policy_metrics,
        "selection_summaries": selection_summaries,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": True,
    }
    payload = {
        "summary": summary,
        "loss_curves": {"stage": config["stage"], "loss_curves": loss_curves},
        "policy_comparison": {
            "stage": config["stage"],
            "policy_metrics": policy_metrics,
            **comparison,
        },
        "context_utility_decision": {
            "stage": config["stage"],
            **comparison,
            "recommended_step30": _recommended_step30(comparison["context_utility_sanity_signal"]),
            "do_not_train_selector_yet": True,
            "do_not_train_current_importance_yet": True,
            "never_claim_final_utility": True,
        },
    }
    _write_outputs(paths, payload)
    return payload


def _policy_dataset(samples: list[dict[str, Any]], policy: dict[str, Any], device: torch.device) -> dict[str, Any]:
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
        target = sample["future_tokens"].detach().to(dtype=torch.float32).mean(dim=(0, 1))
        current_summaries.append(current_summary)
        context_summaries.append(context_summary)
        targets.append(target)
        masses.append(float(selection["selected_importance_mass"]))
        selection_summaries.append({"sample_id": str(sample["sample_id"]), **summarize_selection(selection)})
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
    return {
        "step": int(step),
        "train_loss": _evaluate_loss(model, train_data),
        "val_loss": _evaluate_loss(model, val_data),
    }


def _evaluate_loss(model: BridgeDataContextBottleneckSmokePredictor, data: dict[str, Any]) -> float:
    model.eval()
    with torch.no_grad():
        pred = model.forward_from_summaries(data["current_summaries"], data["context_summaries"])
        loss = F.mse_loss(pred, data["targets"], reduction="mean")
    return float(loss.item())


def _optimizer_scope(model: BridgeDataContextBottleneckSmokePredictor, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    model_param_ids = {id(param) for param in model.parameters()}
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group.get("params", [])
        if isinstance(param, torch.nn.Parameter)
    }
    return {
        "optimizer_step_scope": OPTIMIZER_SCOPE
        if not (optimizer_param_ids - model_param_ids)
        else "external_parameters_present",
        "optimizer_param_count": len(optimizer_param_ids),
    }


def _recommended_step30(signal: str) -> dict[str, str]:
    if signal == "positive":
        return {
            "name": "trained-predictor occlusion teacher on expanded train split",
            "condition": "context utility sanity shows a positive signal",
            "scope": "no selector training yet; no current importance training yet",
        }
    if signal == "negative_or_current_dominant":
        return {
            "name": "diagnose horizon/current dominance before selector training",
            "condition": "context policies do not beat current_only on held-out validation",
            "scope": "reconsider context objective and horizon before selector training",
        }
    return {
        "name": "improve teacher label or increase window diversity before selector training",
        "condition": "Step29 context utility sanity is inconclusive",
        "scope": "no selector training yet; no current importance training yet",
    }


def _missing_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["output"]["split_json"],
        config["output"]["token_manifest_jsonl"],
        config["output"]["token_summary_json"],
        config["output"]["token_smoke_dir"],
        config["output"]["importance_manifest_jsonl"],
        config["output"]["importance_summary_json"],
        config["output"]["importance_dir"],
    ]
    return [str(path) for path in required if not Path(path).exists()]


def _safe_stop_payload(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    summary = {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "tiny_trainval_training_performed": False,
        "training_performed": False,
        "num_selected_windows": 0,
        "num_train_windows": 0,
        "num_val_windows": 0,
        "policies_trained": [],
        "optimizer_step_performed": False,
        "optimizer_step_scope": OPTIMIZER_SCOPE,
        "all_val_losses_finite": False,
        "context_utility_sanity_signal": "safe_stop",
        "context_utility_claim_allowed": False,
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
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }
    return {
        "summary": summary,
        "loss_curves": {"stage": config["stage"], "loss_curves": []},
        "policy_comparison": {"stage": config["stage"], "policy_metrics": [], "context_utility_claim_allowed": False},
        "context_utility_decision": {
            "stage": config["stage"],
            "context_utility_sanity_signal": "safe_stop",
            "context_utility_claim_allowed": False,
            "reason": reason,
            "recommended_step30": _recommended_step30("safe_stop"),
        },
    }


def _write_outputs(paths: dict[str, Path], payload: dict[str, Any]) -> None:
    _write_json(paths["trainval_summary_json"], payload["summary"])
    _write_json(paths["loss_curves_json"], payload["loss_curves"])
    _write_json(paths["policy_comparison_json"], payload["policy_comparison"])
    _write_json(paths["context_utility_decision_json"], payload["context_utility_decision"])


def _output_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {key: Path(value) for key, value in config["output"].items() if key.endswith("_json")}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _finite_metric(item: dict[str, Any], key: str) -> bool:
    return math.isfinite(float(item.get(key, math.nan)))


def _env_guard() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def _resolve_device(requested: str, allow_cpu_fallback: bool) -> torch.device:
    if requested == "cuda_if_available" and torch.cuda.is_available():
        try:
            torch.empty((1,), device="cuda")
            return torch.device("cuda")
        except Exception:
            if not allow_cpu_fallback:
                raise
    if requested.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.empty((1,), device=requested)
            return torch.device(requested)
        except Exception:
            if not allow_cpu_fallback:
                raise
    return torch.device("cpu")


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
    payload = train_eval_step29_context_utility(args.config)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

