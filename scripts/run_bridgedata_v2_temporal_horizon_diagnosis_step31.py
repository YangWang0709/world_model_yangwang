"""Run Step31 temporal/horizon and current-dominance diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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

from analysis.bridgedata_v2_current_dominance_diagnosis import diagnose_current_dominance
from analysis.bridgedata_v2_temporal_horizon_diagnosis import summarize_target_row, summarize_temporal_horizon_results
from data.bridgedata_v2_tfds_world_model_batch import load_bridge_tfds_world_model_samples, validate_world_model_sample
from models.bridgedata_v2_teacher_diagnostic_variants import (
    DIAGNOSTIC_OPTIMIZER_SCOPE,
    BridgeDataTeacherDiagnosticPredictor,
    diagnostic_summaries,
    optimizer_scope_for_diagnostic_predictor,
    target_summary,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_teacher_temporal_horizon_diagnosis_step31.yaml"
POLICY_TO_VARIANT = {
    "current_only": "current_only_predictor",
    "proxy_importance_topk": "current_plus_proxy_topk_context_predictor",
    "full_context_reference": "current_plus_mean_context_predictor",
}


def run_step31_temporal_horizon_diagnosis(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    env_guard = _env_guard()
    missing = _missing_inputs(config)
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"] or missing:
        reason = "env_isaaclab is polluted with TensorFlow/TFDS" if not missing else f"missing Step31 inputs: {missing}"
        payload = _safe_stop_payload(config, reason, env_guard)
        _write_json(Path(config["output"]["temporal_horizon_diagnosis_json"]), payload["temporal_horizon_diagnosis"])
        _write_json(Path(config["output"]["current_dominance_diagnosis_json"]), payload["current_dominance_diagnosis"])
        return payload

    samples = _load_samples(config)
    splits = json.loads(Path(config["input"]["multiseed_splits_json"]).read_text(encoding="utf-8"))
    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    horizon_cfg = config["temporal_horizon_diagnosis"]
    split_seeds = {int(seed) for seed in horizon_cfg.get("split_seeds", [42, 123, 999])}
    target_rows = []
    optimizer_step_performed = False
    optimizer_param_count = 0

    for target_variant in horizon_cfg.get("target_variants", []):
        per_seed_rows = []
        for split in splits.get("splits", []):
            split_seed = int(split["split_seed"])
            if split_seed not in split_seeds:
                continue
            train_samples = [samples_by_id[item] for item in split["train_sample_ids"] if item in samples_by_id]
            val_samples = [samples_by_id[item] for item in split["val_sample_ids"] if item in samples_by_id]
            row: dict[str, Any] = {"split_seed": split_seed}
            for policy_index, policy in enumerate(horizon_cfg.get("compare_policies", [])):
                metric, performed, param_count = _train_policy(config, str(policy), str(target_variant), train_samples, val_samples, split_seed + policy_index)
                row[str(policy)] = metric["val_final_loss"]
                row[f"{policy}_val_best"] = metric["val_best_loss"]
                optimizer_step_performed = optimizer_step_performed or performed
                optimizer_param_count = max(optimizer_param_count, param_count)
            per_seed_rows.append(row)
        target_rows.append(summarize_target_row(str(target_variant), per_seed_rows))

    temporal = summarize_temporal_horizon_results(target_rows)
    temporal.update(
        {
            "stage": config["stage"],
            "safe_stop": False,
            "diagnostic_predictor_training_performed": bool(optimizer_step_performed),
            "optimizer_step_scope": DIAGNOSTIC_OPTIMIZER_SCOPE,
            "optimizer_param_count": int(optimizer_param_count),
            "checkpoint_saved": False,
            "proxy_importance_regeneration_performed": False,
            "teacher_label_regeneration_performed": False,
            "new_tfds_shard_downloaded": False,
            "model_download_performed": False,
            "videomae_training_performed": False,
            "selector_training_performed": False,
            "current_importance_training_performed": False,
            "context_utility_claim_allowed": False,
            "selector_training_allowed": False,
            "current_importance_training_allowed": False,
            "env_isaaclab_guard": env_guard,
        }
    )
    current = diagnose_current_dominance(temporal)
    current.update({"stage": config["stage"], "env_isaaclab_guard": env_guard})
    _write_json(Path(config["output"]["temporal_horizon_diagnosis_json"]), temporal)
    _write_json(Path(config["output"]["current_dominance_diagnosis_json"]), current)
    return {"temporal_horizon_diagnosis": temporal, "current_dominance_diagnosis": current}


def _train_policy(
    config: dict[str, Any],
    policy: str,
    target_variant: str,
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], bool, int]:
    cfg = config["temporal_horizon_diagnosis"]
    variant = POLICY_TO_VARIANT[policy]
    _seed_everything(seed)
    model = BridgeDataTeacherDiagnosticPredictor(
        variant=variant,
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        topk=int(cfg.get("topk", 256)),
        seed=seed,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 0.001)), weight_decay=0.0)
    scope = optimizer_scope_for_diagnostic_predictor(model, optimizer)
    if scope["optimizer_step_scope"] != DIAGNOSTIC_OPTIMIZER_SCOPE:
        raise RuntimeError(f"optimizer scope must be {DIAGNOSTIC_OPTIMIZER_SCOPE}: {scope}")
    train_data = _summary_dataset(train_samples, model, variant, target_variant)
    val_data = _summary_dataset(val_samples, model, variant, target_variant)
    curve = [_curve_point(0, model, train_data, val_data)]
    performed = False
    train_steps = int(cfg.get("train_steps", 160))
    eval_every = max(1, int(cfg.get("eval_every", 40)))
    for step in range(1, train_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model.forward_from_summaries(train_data["current"], train_data["context"])
        loss = F.mse_loss(pred, train_data["target"], reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        performed = True
        if step == 1 or step % eval_every == 0 or step == train_steps:
            curve.append(_curve_point(step, model, train_data, val_data))
    metric = _summarize_curve(policy, target_variant, curve)
    metric.update(
        {
            "optimizer_step_performed": bool(performed),
            "optimizer_step_scope": DIAGNOSTIC_OPTIMIZER_SCOPE,
            "optimizer_param_count": int(scope["optimizer_param_count"]),
            "current_tokens_kept_full": True,
            "train_current_importance": False,
        }
    )
    return metric, performed, int(scope["optimizer_param_count"])


def _summary_dataset(
    samples: list[dict[str, Any]],
    model: BridgeDataTeacherDiagnosticPredictor,
    variant: str,
    target_variant: str,
) -> dict[str, torch.Tensor]:
    current_parts = []
    context_parts = []
    target_parts = []
    with torch.no_grad():
        for sample in samples:
            context = sample["context_tokens"].unsqueeze(0)
            current = sample["current_tokens"].unsqueeze(0)
            importance = sample["context_importance"].unsqueeze(0)
            current_summary, context_summary = diagnostic_summaries(
                context,
                current,
                importance,
                variant=variant,
                topk=model.topk,
                score_vector=model.score_vector.detach(),
            )
            current_parts.append(current_summary.squeeze(0))
            context_parts.append(context_summary.squeeze(0))
            target_parts.append(target_summary(sample["future_tokens"].unsqueeze(0), current, target_variant).squeeze(0))
    return {
        "current": torch.stack(current_parts, dim=0),
        "context": torch.stack(context_parts, dim=0),
        "target": torch.stack(target_parts, dim=0),
    }


def _curve_point(step: int, model: BridgeDataTeacherDiagnosticPredictor, train_data: dict[str, torch.Tensor], val_data: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {"step": int(step), "train_loss": _loss(model, train_data), "val_loss": _loss(model, val_data)}


def _loss(model: BridgeDataTeacherDiagnosticPredictor, data: dict[str, torch.Tensor]) -> float:
    model.eval()
    with torch.no_grad():
        pred = model.forward_from_summaries(data["current"], data["context"])
        loss = F.mse_loss(pred, data["target"], reduction="mean")
    return float(loss.item())


def _summarize_curve(policy: str, target_variant: str, curve: list[dict[str, Any]]) -> dict[str, Any]:
    train_initial = float(curve[0]["train_loss"])
    train_final = float(curve[-1]["train_loss"])
    val_initial = float(curve[0]["val_loss"])
    val_final = float(curve[-1]["val_loss"])
    return {
        "policy": policy,
        "target_variant": target_variant,
        "train_initial_loss": train_initial,
        "train_final_loss": train_final,
        "train_best_loss": min(float(item["train_loss"]) for item in curve),
        "val_initial_loss": val_initial,
        "val_final_loss": val_final,
        "val_best_loss": min(float(item["val_loss"]) for item in curve),
        "val_loss_finite": all(math.isfinite(float(item["val_loss"])) for item in curve),
        "num_curve_points": len(curve),
    }


def _load_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    loader_config = {
        "input": {
            "token_manifest_jsonl": config["input"]["token_manifest_jsonl"],
            "token_summary_json": config["input"]["token_summary_json"],
            "token_smoke_dir": config["input"]["token_smoke_dir"],
            "importance_manifest_jsonl": config["input"]["proxy_importance_manifest_jsonl"],
            "importance_summary_json": config["input"]["proxy_importance_summary_json"],
            "importance_smoke_dir": config["input"]["proxy_importance_dir"],
        },
        "sample_limits": {"max_samples": int(config["data"].get("num_windows_expected", 64))},
    }
    samples = load_bridge_tfds_world_model_samples(loader_config)
    for sample in samples:
        validate_world_model_sample(sample)
    return samples


def _missing_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["input"]["token_manifest_jsonl"],
        config["input"]["token_summary_json"],
        config["input"]["token_smoke_dir"],
        config["input"]["proxy_importance_manifest_jsonl"],
        config["input"]["proxy_importance_summary_json"],
        config["input"]["proxy_importance_dir"],
        config["input"]["multiseed_splits_json"],
        config["input"]["teacher_topk_utility_json"],
    ]
    return [str(path) for path in required if not Path(path).exists()]


def _safe_stop_payload(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    common = {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "token_extraction_performed": False,
        "proxy_importance_regeneration_performed": False,
        "teacher_label_regeneration_performed": False,
        "new_tfds_shard_downloaded": False,
        "model_download_performed": False,
        "videomae_training_performed": False,
        "selector_training_performed": False,
        "current_importance_training_performed": False,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }
    return {
        "temporal_horizon_diagnosis": {**common, "temporal_horizon_diagnosis_performed": False, "target_rows": []},
        "current_dominance_diagnosis": {**common, "current_dominance_diagnosis_performed": False, "current_dominance_level": "unknown"},
    }


def _env_guard() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))


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
    payload = run_step31_temporal_horizon_diagnosis(args.config)
    print(json.dumps(payload["temporal_horizon_diagnosis"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
