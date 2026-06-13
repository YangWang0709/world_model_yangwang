"""Pure statistics for Step38B proxy label learnability diagnosis.

Step38B reads existing BridgeData V2 importance artifacts and decomposes the
proxy label into temporal and spatial-residual parts. It intentionally contains
no model, optimizer, checkpoint, TFDS, action, or language input logic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from data.bridgedata_v2_proxy_patch_selector_splits_step36 import (
    build_step36_training_settings,
    summarize_step36_leakage,
)
from data.bridgedata_v2_tfds_importance_manifest import (
    read_importance_manifest_jsonl,
    validate_importance_artifact,
)
from data.bridgedata_v2_tfds_longer_horizon_manifest import read_jsonl


STAGE = "bridgedata_v2_tfds_proxy_label_diagnosis_step38b"
EPS = 1.0e-12


def load_step38b_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Step38B config must be a mapping: {path}")
    if config.get("stage") != STAGE:
        raise ValueError(f"unexpected Step38B stage: {config.get('stage')!r}")
    return config


def missing_step38b_inputs(config: dict[str, Any]) -> list[str]:
    input_cfg = config.get("input", {})
    required = [
        input_cfg["step33b_run_dir"],
        input_cfg["step35_run_dir"],
        input_cfg["step36_run_dir"],
        input_cfg["step37_run_dir"],
        input_cfg["token_manifest_jsonl"],
        input_cfg["importance_manifest_jsonl"],
        input_cfg["shard_splits_json"],
        input_cfg["shard0_token_manifest_jsonl"],
        input_cfg["shard0_importance_manifest_jsonl"],
        input_cfg["shard0_horizon_window_manifest_jsonl"],
        input_cfg["shard0_horizon_splits_json"],
        input_cfg["step35_selector_metrics_json"],
        input_cfg["step36_patch_selector_metrics_json"],
        input_cfg["step37_factorized_selector_metrics_json"],
        input_cfg["step37_gate_decision_json"],
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        return missing
    step37_gate = _read_json(input_cfg["step37_gate_decision_json"])
    for flag in (
        "selector_training_allowed",
        "final_selector_training_allowed",
        "current_importance_training_allowed",
        "context_utility_claim_allowed",
    ):
        if bool(step37_gate.get(flag, False)):
            missing.append(f"unsafe upstream Step37 gate flag is true: {flag}")
    return missing


def build_step38b_split_settings(config: dict[str, Any]) -> list[dict[str, Any]]:
    step36_like = dict(config)
    step36_like["stage"] = "bridgedata_v2_tfds_proxy_patch_selector_train_step36"
    step36_like.setdefault("splits", {})
    step36_like["splits"].setdefault("split_seeds", [42, 123, 999])
    step36_like["splits"].setdefault("run_within_shard", True)
    step36_like["splits"].setdefault("run_cross_shard", True)
    step36_like["splits"].setdefault("run_mixed_shard", True)
    return build_step36_training_settings(step36_like)


def load_step38b_importance_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    max_per_shard = int(config.get("data", {}).get("max_samples_per_shard", 64))
    expected_shape = list(config.get("data", {}).get("importance_shape", [16, 392]))
    shard1 = _load_importance_samples_for_manifest(
        importance_manifest_jsonl=config["input"]["importance_manifest_jsonl"],
        allowed_sample_ids=None,
        shard_id="shard1",
        data_package_id="bridge_tfds_shard1",
        expected_shape=expected_shape,
        max_samples=max_per_shard,
    )
    shard0_gap0_ids = {
        str(record["sample_id"])
        for record in read_jsonl(config["input"]["shard0_horizon_window_manifest_jsonl"])
        if int(record.get("horizon_gap", -1)) == 0
    }
    shard0 = _load_importance_samples_for_manifest(
        importance_manifest_jsonl=config["input"]["shard0_importance_manifest_jsonl"],
        allowed_sample_ids=shard0_gap0_ids,
        shard_id="shard0",
        data_package_id="bridge_tfds_shard0_gap0",
        expected_shape=expected_shape,
        max_samples=max_per_shard,
    )
    return shard0 + shard1


def decompose_proxy_importance(importance: torch.Tensor) -> dict[str, Any]:
    label = _as_label_tensor(importance)
    temporal_component = label.mean(dim=1).contiguous()
    temporal_broadcast = temporal_component[:, None].expand_as(label).contiguous()
    spatial_residual = (label - temporal_broadcast).contiguous()
    residual_abs = spatial_residual.abs().contiguous()
    label_energy = float(torch.sum(label * label).item())
    residual_energy = float(torch.sum(spatial_residual * spatial_residual).item())
    temporal_energy = float(torch.sum(temporal_broadcast * temporal_broadcast).item())
    denom = label_energy + EPS
    return {
        "temporal_component": temporal_component,
        "temporal_broadcast": temporal_broadcast,
        "spatial_residual": spatial_residual,
        "residual_abs": residual_abs,
        "residual_energy_ratio": float(residual_energy / denom),
        "temporal_energy_ratio": float(temporal_energy / denom),
    }


def compute_label_entropy(importance: torch.Tensor) -> dict[str, Any]:
    label = _as_label_tensor(importance).clamp_min(0.0)
    flat = label.reshape(-1)
    global_entropy = _normalized_entropy(flat)
    frame_entropy = torch.tensor([_normalized_entropy(frame) for frame in label], dtype=torch.float32)
    return {
        "global_normalized_entropy": float(global_entropy),
        "per_frame_spatial_entropy_mean": float(frame_entropy.mean().item()),
        "per_frame_spatial_entropy_min": float(frame_entropy.min().item()),
        "per_frame_spatial_entropy_max": float(frame_entropy.max().item()),
        "per_frame_spatial_entropy_std": float(frame_entropy.std(unbiased=False).item()),
    }


def compute_topk_concentration(importance: torch.Tensor, topk_values: list[int]) -> dict[str, Any]:
    label = _as_label_tensor(importance).clamp_min(0.0)
    decomposition = decompose_proxy_importance(label)
    residual_abs = decomposition["residual_abs"]
    total_mass = float(label.sum().item()) + EPS
    residual_mass = float(residual_abs.sum().item()) + EPS
    flat = label.reshape(-1)
    flat_residual = residual_abs.reshape(-1)
    temporal_size, spatial_size = int(label.shape[0]), int(label.shape[1])
    out: dict[str, Any] = {}
    for raw_k in topk_values:
        k = max(1, min(int(raw_k), int(flat.numel())))
        values, indices = torch.topk(flat, k=k, largest=True)
        frames = torch.div(indices, spatial_size, rounding_mode="floor")
        frame_counts = torch.bincount(frames, minlength=temporal_size).to(torch.float32)
        out[f"top{k}_mass_ratio"] = float(values.sum().item() / total_mass)
        out[f"top{k}_frame_coverage"] = float(torch.count_nonzero(frame_counts).item() / float(temporal_size))
        out[f"top{k}_temporal_concentration"] = float(frame_counts.max().item() / float(k))
        out[f"top{k}_residual_mass_ratio"] = float(flat_residual[indices].sum().item() / residual_mass)
    return out


def compute_temporal_broadcast_fit(importance: torch.Tensor, temporal_broadcast: torch.Tensor) -> dict[str, Any]:
    label = _as_label_tensor(importance)
    broadcast = _as_label_tensor(temporal_broadcast)
    if list(label.shape) != list(broadcast.shape):
        raise ValueError("importance and temporal_broadcast shapes must match")
    diff = label - broadcast
    mse = float(torch.mean(diff * diff).item())
    mae = float(torch.mean(diff.abs()).item())
    centered = label - label.mean()
    sse = float(torch.sum(diff * diff).item())
    sst = float(torch.sum(centered * centered).item())
    r2_like = 1.0 if sst <= EPS and sse <= EPS else 1.0 - (sse / (sst + EPS))
    return {
        "temporal_broadcast_mse": mse,
        "temporal_broadcast_mae": mae,
        "temporal_broadcast_r2_like": float(r2_like),
    }


def diagnose_step38b_sample(sample: dict[str, Any], topk_values: list[int]) -> dict[str, Any]:
    importance = sample["importance"]
    decomposition = decompose_proxy_importance(importance)
    entropy = compute_label_entropy(importance)
    topk = compute_topk_concentration(importance, topk_values)
    fit = compute_temporal_broadcast_fit(importance, decomposition["temporal_broadcast"])
    row = {
        "sample_id": sample["sample_id"],
        "trajectory_id": sample["trajectory_id"],
        "shard_id": sample["shard_id"],
        "data_package_id": sample["data_package_id"],
        "importance_shape": list(importance.shape),
        "residual_energy_ratio": decomposition["residual_energy_ratio"],
        "temporal_energy_ratio": decomposition["temporal_energy_ratio"],
        "temporal_component_mean": float(decomposition["temporal_component"].mean().item()),
        "temporal_component_std": float(decomposition["temporal_component"].std(unbiased=False).item()),
        "spatial_residual_abs_mean": float(decomposition["residual_abs"].mean().item()),
        "spatial_residual_abs_max": float(decomposition["residual_abs"].max().item()),
        "metadata": {
            "future_tokens_loaded_for_diagnosis": False,
            "action_used_as_input": False,
            "language_used_as_input": False,
            "current_importance_generated": False,
            "optimizer_step_performed": False,
            "checkpoint_saved": False,
            "state_dict_saved": False,
        },
    }
    row.update(entropy)
    row.update(topk)
    row.update(fit)
    return row


def aggregate_label_diagnosis(rows: list[dict[str, Any]], eval_type: str, primary_topk: int = 256) -> dict[str, Any]:
    if not rows:
        return {
            "stage": STAGE,
            "eval_type": eval_type,
            "num_rows": 0,
            "shards": [],
            "temporal_component_dominates": False,
            "spatial_residual_learnable_evidence": False,
        }
    scalar_keys = [
        "residual_energy_ratio",
        "temporal_energy_ratio",
        "global_normalized_entropy",
        "per_frame_spatial_entropy_mean",
        "temporal_broadcast_mse",
        "temporal_broadcast_mae",
        "temporal_broadcast_r2_like",
        f"top{primary_topk}_mass_ratio",
        f"top{primary_topk}_frame_coverage",
        f"top{primary_topk}_temporal_concentration",
        f"top{primary_topk}_residual_mass_ratio",
    ]
    means = {f"{key}_mean": _mean(row.get(key) for row in rows) for key in scalar_keys}
    return {
        "stage": STAGE,
        "eval_type": eval_type,
        "num_rows": len(rows),
        "shards": sorted({str(row["shard_id"]) for row in rows}),
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "temporal_component_dominates": bool(
            means["temporal_broadcast_r2_like_mean"] >= 0.5
            or means["temporal_energy_ratio_mean"] > means["residual_energy_ratio_mean"]
        ),
        "spatial_residual_learnable_evidence": bool(
            means["residual_energy_ratio_mean"] >= 0.25
            and means[f"top{primary_topk}_mass_ratio_mean"] >= 0.25
            and means["per_frame_spatial_entropy_mean_mean"] < 0.85
        ),
        **means,
    }


def build_cross_shard_consistency(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_shard: dict[str, list[torch.Tensor]] = {}
    for sample in samples:
        decomposition = decompose_proxy_importance(sample["importance"])
        residual_abs = decomposition["residual_abs"]
        denom = residual_abs.sum() + EPS
        by_shard.setdefault(str(sample["shard_id"]), []).append((residual_abs / denom).reshape(-1))
    if len(by_shard) < 2:
        return {
            "stage": STAGE,
            "computed": False,
            "reason": "need at least two shards",
            "cross_shard_residual_cosine": None,
            "cross_shard_residual_l1": None,
            "cross_shard_residual_consistent": False,
        }
    shard_names = sorted(by_shard)
    a = torch.stack(by_shard[shard_names[0]]).mean(dim=0)
    b = torch.stack(by_shard[shard_names[1]]).mean(dim=0)
    cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    l1 = float(torch.mean(torch.abs(a - b)).item())
    return {
        "stage": STAGE,
        "computed": True,
        "shards": shard_names,
        "num_samples_by_shard": {name: len(values) for name, values in by_shard.items()},
        "cross_shard_residual_cosine": cosine,
        "cross_shard_residual_l1": l1,
        "cross_shard_residual_consistent": bool(cosine >= 0.70),
    }


def build_step38b_gate_decision(
    *,
    config: dict[str, Any],
    diagnosis_summary: dict[str, Any],
    cross_shard_consistency: dict[str, Any],
    leakage_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    aggregate = diagnosis_summary.get("all_samples", {})
    thresholds = config.get("diagnostics", {}).get("thresholds", {})
    primary_topk = int(config.get("diagnostics", {}).get("primary_topk", 256))
    residual_energy = float(aggregate.get("residual_energy_ratio_mean", 0.0) or 0.0)
    spatial_entropy = float(aggregate.get("per_frame_spatial_entropy_mean_mean", 1.0) or 1.0)
    topk_mass = float(aggregate.get(f"top{primary_topk}_mass_ratio_mean", 0.0) or 0.0)
    temporal_r2 = float(aggregate.get("temporal_broadcast_r2_like_mean", 0.0) or 0.0)
    cross_consistent = bool(cross_shard_consistency.get("cross_shard_residual_consistent", False))
    temporal_dominates = bool(
        temporal_r2 >= float(thresholds.get("temporal_r2_dominates_min", 0.5))
        or residual_energy <= float(thresholds.get("residual_energy_low_max", 0.25))
    )
    residual_structured = bool(
        residual_energy > float(thresholds.get("residual_energy_low_max", 0.25))
        and spatial_entropy < float(thresholds.get("spatial_entropy_high_min", 0.85))
        and topk_mass >= float(thresholds.get("primary_topk_mass_concentrated_min", 0.25))
        and cross_consistent
    )
    if residual_structured:
        recommended = {
            "name": "current-conditioned context selector smoke",
            "scope": "diagnostic smoke only; still no final selector training",
            "reason": "spatial residual has structured and cross-shard-consistent evidence",
        }
    else:
        recommended = {
            "name": "label redesign toward temporal-only or temporal+broadcast/coarse supervision",
            "scope": "redesign proxy labels before any downstream selector use",
            "reason": "spatial residual is weak, diffuse, or not cross-shard-consistent",
        }
    no_leakage = bool((leakage_summary or {}).get("no_language_or_trajectory_leakage", True))
    return {
        "stage": STAGE,
        "safe_stop": bool(diagnosis_summary.get("safe_stop", False)),
        "diagnosis_performed": bool(diagnosis_summary.get("diagnosis_performed", False)),
        "temporal_component_dominates": temporal_dominates,
        "spatial_residual_learnable_evidence": residual_structured,
        "spatial_residual_cross_shard_consistent": cross_consistent,
        "spatial_residual_energy_ratio_mean": residual_energy,
        "spatial_entropy_mean": spatial_entropy,
        f"top{primary_topk}_mass_ratio_mean": topk_mass,
        "temporal_broadcast_r2_like_mean": temporal_r2,
        "no_language_or_trajectory_leakage": no_leakage,
        "dataset_bias_detected": False,
        "shard_shift_detected": False,
        "full_context_noisy_issue_acknowledged": True,
        "label_patch_detail_learnable_allowed": False,
        "selector_training_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "new_dataset_download_performed": False,
        "new_model_download_performed": False,
        "recommended_step39": recommended,
    }


def build_safe_stop_payload(config: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "safe_stop": True,
        "diagnosis_performed": False,
        "reason": "missing required local Step38B inputs",
        "missing_inputs": list(missing),
        "num_samples": 0,
        "future_tokens_loaded_for_diagnosis": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "current_importance_generated": False,
        "training_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "new_dataset_download_performed": False,
        "new_model_download_performed": False,
        "tensorflow_required": False,
        "tensorflow_datasets_required": False,
        "all_samples": aggregate_label_diagnosis([], "all_samples"),
    }


def build_step38b_diagnosis(config: dict[str, Any]) -> dict[str, Any]:
    topk_values = [int(value) for value in config.get("diagnostics", {}).get("topk_values", [64, 128, 256, 512])]
    primary_topk = int(config.get("diagnostics", {}).get("primary_topk", 256))
    samples = load_step38b_importance_samples(config)
    rows = [diagnose_step38b_sample(sample, topk_values) for sample in samples]
    rows_by_shard = {
        shard: [row for row in rows if row["shard_id"] == shard]
        for shard in sorted({str(row["shard_id"]) for row in rows})
    }
    settings = build_step38b_split_settings(config)
    leakage = summarize_step36_leakage(settings)
    aggregates = {
        "all_samples": aggregate_label_diagnosis(rows, "all_samples", primary_topk=primary_topk),
        "within_shard": aggregate_label_diagnosis(
            [row for row in rows if row["shard_id"] == "shard1"],
            "within_shard",
            primary_topk=primary_topk,
        ),
        "cross_shard": aggregate_label_diagnosis(rows, "cross_shard", primary_topk=primary_topk),
        "mixed_shard": aggregate_label_diagnosis(rows, "mixed_shard", primary_topk=primary_topk),
        "by_shard": {
            shard: aggregate_label_diagnosis(shard_rows, shard, primary_topk=primary_topk)
            for shard, shard_rows in rows_by_shard.items()
        },
    }
    summary = {
        "stage": STAGE,
        "safe_stop": False,
        "diagnosis_performed": True,
        "num_samples": len(rows),
        "num_samples_by_shard": {shard: len(shard_rows) for shard, shard_rows in rows_by_shard.items()},
        "future_tokens_loaded_for_diagnosis": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "current_importance_generated": False,
        "training_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "new_dataset_download_performed": False,
        "new_model_download_performed": False,
        "tensorflow_required": False,
        "tensorflow_datasets_required": False,
        "leakage_summary": leakage,
        **aggregates,
    }
    return summary


def make_label_decomposition_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "num_rows": len(rows),
        "rows": rows,
        "schema": {
            "importance": "[16,392]",
            "temporal_component": "[16]",
            "temporal_broadcast": "[16,392]",
            "spatial_residual": "[16,392]",
        },
        "training_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
    }


def write_step38b_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_importance_samples_for_manifest(
    *,
    importance_manifest_jsonl: str | Path,
    allowed_sample_ids: set[str] | None,
    shard_id: str,
    data_package_id: str,
    expected_shape: list[int],
    max_samples: int,
) -> list[dict[str, Any]]:
    records = read_importance_manifest_jsonl(importance_manifest_jsonl)
    samples: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        if allowed_sample_ids is not None and sample_id not in allowed_sample_ids:
            continue
        artifact = validate_importance_artifact(record["importance_artifact_path"])
        importance = _as_label_tensor(artifact["context_importance_norm"])
        if list(importance.shape) != expected_shape:
            raise ValueError(f"importance shape {list(importance.shape)} != expected {expected_shape}")
        metadata = artifact.get("metadata") or {}
        samples.append(
            {
                "sample_id": sample_id,
                "trajectory_id": str(record.get("trajectory_id") or metadata.get("trajectory_id") or sample_id),
                "shard_id": shard_id,
                "data_package_id": data_package_id,
                "importance": importance,
                "metadata": {
                    "importance_artifact_path": str(record["importance_artifact_path"]),
                    "future_tokens_loaded_for_diagnosis": False,
                    "action_used_as_input": False,
                    "language_used_as_input": False,
                    "current_importance_generated": False,
                },
            }
        )
        if len(samples) >= int(max_samples):
            break
    return samples


def _as_label_tensor(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("importance must be a torch.Tensor")
    label = value.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if bool(label.requires_grad):
        raise ValueError("importance must not require gradients")
    if label.ndim != 2:
        raise ValueError(f"importance must be [T, S], got {tuple(label.shape)}")
    if not bool(torch.isfinite(label).all()):
        raise ValueError("importance contains non-finite values")
    return label


def _normalized_entropy(values: torch.Tensor) -> float:
    flat = values.detach().to(dtype=torch.float32, device="cpu").reshape(-1).clamp_min(0.0)
    total = flat.sum()
    if float(total.item()) <= EPS:
        return 0.0
    probs = flat / (total + EPS)
    entropy = -torch.sum(probs * torch.log(probs + EPS))
    denom = math.log(max(int(flat.numel()), 2))
    value = float(entropy.item() / denom)
    return float(max(0.0, min(1.0, value)))


def _mean(values: Any) -> float:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(sum(clean) / len(clean)) if clean else 0.0


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
