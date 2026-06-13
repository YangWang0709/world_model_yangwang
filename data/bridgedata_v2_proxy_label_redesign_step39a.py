"""Step39A proxy-label redesign diagnostics.

This module builds alternative labels from existing Step33B/Step38B importance
artifacts. It is pure tensor/statistics code: no optimizer, no model training,
no checkpoint writing, no TFDS access, and no action/language/future-token input.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from data.bridgedata_v2_proxy_label_diagnosis_step38b import (
    compute_label_entropy,
    compute_temporal_broadcast_fit,
    compute_topk_concentration,
    decompose_proxy_importance,
    load_step38b_importance_samples,
)


STAGE = "bridgedata_v2_tfds_proxy_label_redesign_step39a"
EPS = 1.0e-12


def load_step39a_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Step39A config must be a mapping: {path}")
    if config.get("stage") != STAGE:
        raise ValueError(f"unexpected Step39A stage: {config.get('stage')!r}")
    return config


def missing_step39a_inputs(config: dict[str, Any]) -> list[str]:
    input_cfg = config.get("input", {})
    required = [
        input_cfg["step38b_run_dir"],
        input_cfg["step38b_diagnosis_summary_json"],
        input_cfg["step38b_label_decomposition_json"],
        input_cfg["step38b_cross_shard_consistency_json"],
        input_cfg["step38b_gate_decision_json"],
        input_cfg["step33b_run_dir"],
        input_cfg["step37_run_dir"],
        input_cfg["importance_manifest_jsonl"],
        input_cfg["shard0_importance_manifest_jsonl"],
        input_cfg["shard0_horizon_window_manifest_jsonl"],
        input_cfg["step37_gate_decision_json"],
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        return missing
    for label, path in (
        ("Step38B", input_cfg["step38b_gate_decision_json"]),
        ("Step37", input_cfg["step37_gate_decision_json"]),
    ):
        gate = _read_json(path)
        for flag in (
            "selector_training_allowed",
            "final_selector_training_allowed",
            "current_importance_training_allowed",
            "context_utility_claim_allowed",
        ):
            if bool(gate.get(flag, False)):
                missing.append(f"unsafe upstream {label} gate flag is true: {flag}")
    return missing


def load_step39a_importance_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    step38_like = dict(config)
    step38_like["stage"] = "bridgedata_v2_tfds_proxy_label_diagnosis_step38b"
    step38_like["data"] = dict(config.get("data", {}))
    step38_like["data"]["load_future_tokens_for_diagnosis"] = False
    return load_step38b_importance_samples(step38_like)


def build_temporal_only_label(importance: torch.Tensor) -> dict[str, torch.Tensor]:
    label = _label(importance)
    native = label.mean(dim=1).contiguous()
    return {"native_label": native, "broadcast_label": native[:, None].expand_as(label).contiguous()}


def build_temporal_broadcast_label(importance: torch.Tensor) -> torch.Tensor:
    return build_temporal_only_label(importance)["broadcast_label"]


def build_temporal_broadcast_minmax_label(importance: torch.Tensor) -> torch.Tensor:
    native = build_temporal_only_label(importance)["native_label"]
    native = _unit_minmax(native)
    return native[:, None].expand_as(_label(importance)).contiguous()


def build_temporal_broadcast_rank_soft_label(importance: torch.Tensor) -> torch.Tensor:
    native = build_temporal_only_label(importance)["native_label"]
    order = torch.argsort(native, stable=True)
    ranks = torch.empty_like(native)
    ranks[order] = torch.arange(native.numel(), dtype=torch.float32)
    soft = ranks / float(max(int(native.numel()) - 1, 1))
    return soft[:, None].expand_as(_label(importance)).contiguous()


def build_coarse_index_bins_label(importance: torch.Tensor, num_bins: int) -> torch.Tensor:
    label = _label(importance)
    temporal, spatial = int(label.shape[0]), int(label.shape[1])
    if int(num_bins) < 1 or int(num_bins) > spatial:
        raise ValueError("num_bins must be in [1, spatial_tokens]")
    edges = torch.linspace(0, spatial, steps=int(num_bins) + 1).round().to(torch.int64)
    out = torch.zeros_like(label)
    for start, end in zip(edges[:-1], edges[1:]):
        left, right = int(start.item()), int(end.item())
        if right <= left:
            continue
        out[:, left:right] = label[:, left:right].mean(dim=1, keepdim=True).expand(temporal, right - left)
    return out.contiguous()


def build_coarse_grid_label_if_possible(
    importance: torch.Tensor,
    grid_shape: list[int] | tuple[int, int] = (14, 28),
    coarse_shape: list[int] | tuple[int, int] = (7, 14),
) -> torch.Tensor:
    label = _label(importance)
    grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
    coarse_h, coarse_w = int(coarse_shape[0]), int(coarse_shape[1])
    if grid_h * grid_w != int(label.shape[1]):
        raise ValueError("grid_shape must multiply to spatial token count")
    if grid_h % coarse_h != 0 or grid_w % coarse_w != 0:
        raise ValueError("coarse_shape must evenly divide grid_shape")
    block_h, block_w = grid_h // coarse_h, grid_w // coarse_w
    grid = label.reshape(int(label.shape[0]), grid_h, grid_w)
    coarse = grid.reshape(int(label.shape[0]), coarse_h, block_h, coarse_w, block_w).mean(dim=(2, 4))
    expanded = coarse[:, :, None, :, None].expand(int(label.shape[0]), coarse_h, block_h, coarse_w, block_w)
    return expanded.reshape_as(label).contiguous()


def build_denoised_soft_topk_label(
    importance: torch.Tensor,
    k: int,
    temperature: float,
    floor: float,
) -> torch.Tensor:
    label = _unit_minmax(_label(importance))
    flat = label.reshape(-1)
    topk = max(1, min(int(k), int(flat.numel())))
    values, indices = torch.topk(flat, k=topk, largest=True)
    threshold = values.min()
    out = torch.full_like(flat, float(floor))
    soft = torch.sigmoid((flat[indices] - threshold) / max(float(temperature), 1.0e-6))
    out[indices] = float(floor) + (1.0 - float(floor)) * soft
    return out.reshape_as(label).clamp(0.0, 1.0).contiguous()


def build_global_spatial_prior(importance_samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not importance_samples:
        raise ValueError("need samples to build global spatial prior")
    spatial = torch.stack([_label(sample["importance"]).mean(dim=0) for sample in importance_samples]).mean(dim=0)
    prior = _unit_minmax(spatial).contiguous()
    return {
        "prior": prior,
        "stats": {
            "shape": list(prior.shape),
            "mean": float(prior.mean().item()),
            "std": float(prior.std(unbiased=False).item()),
            "min": float(prior.min().item()),
            "max": float(prior.max().item()),
            "top64_mass_ratio": _topk_mass_1d(prior, 64),
            "top256_mass_ratio": _topk_mass_1d(prior, min(256, int(prior.numel()))),
        },
    }


def build_global_spatial_prior_removed_residual_label(importance: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    label = _label(importance)
    prior = _prior(prior, label.shape[1])
    temporal = build_temporal_broadcast_label(label)
    residual = (label - prior[None, :]).clamp_min(0.0)
    return (temporal + residual).clamp(0.0, 1.0).contiguous()


def build_temporal_plus_global_spatial_prior_label(importance: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    label = _label(importance)
    prior = _prior(prior, label.shape[1])
    temporal = build_temporal_broadcast_label(label)
    combined = temporal + prior[None, :]
    return _unit_minmax(combined).contiguous()


def build_label_variants_for_sample(
    importance: torch.Tensor,
    prior: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    variants: dict[str, torch.Tensor] = {}
    enabled = set(config.get("label_variants", {}).get("enabled", []))
    if "temporal_only" in enabled:
        variants["temporal_only"] = build_temporal_only_label(importance)["broadcast_label"]
    if "temporal_broadcast" in enabled:
        variants["temporal_broadcast"] = build_temporal_broadcast_label(importance)
    if "temporal_broadcast_minmax" in enabled:
        variants["temporal_broadcast_minmax"] = build_temporal_broadcast_minmax_label(importance)
    if "temporal_broadcast_rank_soft" in enabled:
        variants["temporal_broadcast_rank_soft"] = build_temporal_broadcast_rank_soft_label(importance)
    for bins in config.get("label_variants", {}).get("coarse_index_bins", [49, 98, 196]):
        name = f"coarse_index_bins_{int(bins)}"
        if name in enabled:
            variants[name] = build_coarse_index_bins_label(importance, int(bins))
    grid_shape = config.get("label_variants", {}).get("optional_grid_shape", [14, 28])
    for coarse_shape in config.get("label_variants", {}).get("coarse_grid_shapes", [[7, 14], [7, 7]]):
        name = f"coarse_grid_{int(coarse_shape[0])}x{int(coarse_shape[1])}_if_{int(grid_shape[0])}x{int(grid_shape[1])}"
        if name in enabled:
            variants[name] = build_coarse_grid_label_if_possible(importance, grid_shape, coarse_shape)
    topk_temperature = float(config.get("label_variants", {}).get("soft_topk_temperature", 0.15))
    topk_floor = float(config.get("label_variants", {}).get("soft_topk_floor", 0.02))
    for k in config.get("label_variants", {}).get("topk_values", [64, 128, 256, 512]):
        name = f"denoised_soft_topk_{int(k)}"
        if name in enabled:
            variants[name] = build_denoised_soft_topk_label(importance, int(k), topk_temperature, topk_floor)
    if "global_spatial_prior_removed_residual" in enabled:
        variants["global_spatial_prior_removed_residual"] = build_global_spatial_prior_removed_residual_label(
            importance, prior
        )
    if "temporal_plus_global_spatial_prior" in enabled:
        variants["temporal_plus_global_spatial_prior"] = build_temporal_plus_global_spatial_prior_label(importance, prior)
    return variants


def compute_label_variant_metrics(
    original: torch.Tensor,
    variant_label: torch.Tensor,
    variant_name: str,
    topk_values: list[int],
) -> dict[str, Any]:
    original = _label(original)
    variant = _label(variant_label)
    if list(original.shape) != list(variant.shape):
        raise ValueError("original and variant labels must have matching [T,S] shapes")
    diff = original - variant
    entropy = compute_label_entropy(variant)
    topk = compute_topk_concentration(variant, topk_values)
    decomp = decompose_proxy_importance(variant)
    fit = compute_temporal_broadcast_fit(variant, decomp["temporal_broadcast"])
    primary_topk = int(topk_values[-2] if 256 in topk_values else topk_values[0])
    metrics = {
        "variant_name": str(variant_name),
        "reconstruction_mse_to_original": float(torch.mean(diff * diff).item()),
        "reconstruction_mae_to_original": float(torch.mean(diff.abs()).item()),
        "temporal_r2_like": float(fit["temporal_broadcast_r2_like"]),
        "residual_energy_after_variant": float(decomp["residual_energy_ratio"]),
        "label_min": float(variant.min().item()),
        "label_max": float(variant.max().item()),
        "label_mean": float(variant.mean().item()),
        "label_std": float(variant.std(unbiased=False).item()),
        "finite": bool(torch.isfinite(variant).all().item()),
        "primary_topk": primary_topk,
    }
    metrics.update(entropy)
    metrics.update(topk)
    return metrics


def aggregate_variant_metrics(rows: list[dict[str, Any]], variant_name: str) -> dict[str, Any]:
    subset = [row for row in rows if row["variant_name"] == variant_name]
    if not subset:
        return {"stage": STAGE, "variant_name": variant_name, "num_rows": 0}
    scalar_keys = [
        "reconstruction_mse_to_original",
        "reconstruction_mae_to_original",
        "temporal_r2_like",
        "residual_energy_after_variant",
        "global_normalized_entropy",
        "per_frame_spatial_entropy_mean",
        "label_min",
        "label_max",
        "label_mean",
        "label_std",
        "top64_mass_ratio",
        "top128_mass_ratio",
        "top256_mass_ratio",
        "top512_mass_ratio",
        "top256_frame_coverage",
        "top256_temporal_concentration",
    ]
    payload = {
        "stage": STAGE,
        "variant_name": variant_name,
        "num_rows": len(subset),
        "finite": all(bool(row.get("finite", False)) for row in subset),
    }
    payload.update({f"{key}_mean": _mean(row.get(key) for row in subset) for key in scalar_keys})
    return payload


def compute_cross_shard_variant_consistency(
    rows: list[dict[str, Any]],
    threshold: float = 0.70,
) -> dict[str, Any]:
    by_variant: dict[str, dict[str, list[torch.Tensor]]] = {}
    for row in rows:
        flat = row.get("_label_flat")
        if not isinstance(flat, torch.Tensor):
            continue
        by_variant.setdefault(str(row["variant_name"]), {}).setdefault(str(row["shard_id"]), []).append(flat)
    results: dict[str, Any] = {}
    for variant_name, by_shard in sorted(by_variant.items()):
        if len(by_shard) < 2:
            results[variant_name] = {
                "computed": False,
                "cross_shard_cosine": None,
                "cross_shard_l1": None,
                "cross_shard_consistent": False,
            }
            continue
        shard_names = sorted(by_shard)
        a = torch.stack(by_shard[shard_names[0]]).mean(dim=0)
        b = torch.stack(by_shard[shard_names[1]]).mean(dim=0)
        cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
        l1 = float(torch.mean(torch.abs(a - b)).item())
        results[variant_name] = {
            "computed": True,
            "shards": shard_names,
            "num_rows_by_shard": {name: len(values) for name, values in by_shard.items()},
            "cross_shard_cosine": cosine,
            "cross_shard_l1": l1,
            "cross_shard_consistent": bool(cosine >= float(threshold)),
        }
    return {"stage": STAGE, "by_variant": results}


def rank_label_variants(
    variant_metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    primary_mass_min = float(thresholds.get("primary_topk_mass_min", 0.25))
    entropy_max = float(thresholds.get("entropy_max_for_patch_like_label", 0.85))
    reconstruction_max = float(thresholds.get("reconstruction_mse_reasonable_max", 0.08))
    residual_max = float(thresholds.get("residual_energy_after_redesign_max", 0.35))
    for name, metrics in sorted(variant_metrics.get("aggregates", {}).items()):
        consistency = variant_metrics.get("cross_shard_consistency", {}).get("by_variant", {}).get(name, {})
        topk = float(metrics.get("top256_mass_ratio_mean", 0.0) or 0.0)
        entropy = float(metrics.get("per_frame_spatial_entropy_mean_mean", 1.0) or 1.0)
        mse = float(metrics.get("reconstruction_mse_to_original_mean", 1.0) or 1.0)
        residual = float(metrics.get("residual_energy_after_variant_mean", 1.0) or 1.0)
        cosine = float(consistency.get("cross_shard_cosine") or 0.0)
        concentration_score = min(1.0, topk / max(primary_mass_min, EPS))
        entropy_score = max(0.0, 1.0 - entropy / max(entropy_max, EPS))
        fidelity_score = max(0.0, 1.0 - mse / max(reconstruction_max, EPS))
        residual_score = max(0.0, 1.0 - residual / max(residual_max, EPS))
        consistency_score = max(0.0, min(1.0, cosine))
        diagnosis_score = (
            0.25 * concentration_score
            + 0.20 * entropy_score
            + 0.20 * consistency_score
            + 0.20 * fidelity_score
            + 0.15 * residual_score
        )
        rows.append(
            {
                "variant_name": name,
                "diagnosis_score": float(diagnosis_score),
                "top256_mass_ratio_mean": topk,
                "per_frame_spatial_entropy_mean_mean": entropy,
                "reconstruction_mse_to_original_mean": mse,
                "residual_energy_after_variant_mean": residual,
                "cross_shard_cosine": cosine,
                "cross_shard_consistent": bool(consistency.get("cross_shard_consistent", False)),
                "score_is_engineering_heuristic": True,
            }
        )
    ranked = sorted(rows, key=lambda row: row["diagnosis_score"], reverse=True)
    return {"stage": STAGE, "ranked_variants": ranked, "best_variant": ranked[0] if ranked else None}


def build_step39a_gate_decision(
    *,
    comparison: dict[str, Any],
    step38b_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    best = comparison.get("ranking", {}).get("best_variant") or {}
    best_name = best.get("variant_name")
    candidate_ready = bool(best_name and float(best.get("diagnosis_score", 0.0)) >= 0.55)
    if candidate_ready:
        step40 = {
            "label_variant": best_name,
            "scope": "bounded selector smoke only; no final selector, current importance, or downstream utility claim",
            "reason": "highest engineering diagnosis score among safer redesigned labels",
        }
    else:
        step40 = {
            "label_variant": "continue_label_redesign",
            "scope": "continue proxy label redesign before any selector smoke",
            "reason": "no redesigned label candidate clearly improved stability and deployability",
        }
    return {
        "stage": STAGE,
        "redesigned_label_candidate_ready": candidate_ready,
        "recommended_step40_label_variant": step40["label_variant"],
        "recommended_step40_scope": step40["scope"],
        "recommended_step40_reason": step40["reason"],
        "upstream_step38b_spatial_residual_learnable_evidence": bool(
            (step38b_gate or {}).get("spatial_residual_learnable_evidence", False)
        ),
        "selector_training_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "training_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "new_dataset_download_performed": False,
        "new_model_download_performed": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "future_tokens_used_as_input": False,
    }


def build_safe_stop_payload(config: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "safe_stop": True,
        "label_redesign_performed": False,
        "reason": "missing required local Step39A inputs",
        "missing_inputs": list(missing),
        "num_samples": 0,
        "training_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "new_dataset_download_performed": False,
        "new_model_download_performed": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "future_tokens_used_as_input": False,
    }


def run_step39a_redesign_statistics(config: dict[str, Any]) -> dict[str, Any]:
    samples = load_step39a_importance_samples(config)
    topk_values = [int(value) for value in config.get("label_variants", {}).get("topk_values", [64, 128, 256, 512])]
    prior_payload = build_global_spatial_prior(samples)
    prior = prior_payload["prior"]
    metric_rows: list[dict[str, Any]] = []
    for sample in samples:
        variants = build_label_variants_for_sample(sample["importance"], prior, config)
        for variant_name, variant_label in variants.items():
            row = compute_label_variant_metrics(sample["importance"], variant_label, variant_name, topk_values)
            row.update(
                {
                    "sample_id": sample["sample_id"],
                    "trajectory_id": sample["trajectory_id"],
                    "shard_id": sample["shard_id"],
                    "data_package_id": sample["data_package_id"],
                    "_label_flat": variant_label.detach().reshape(-1).to(torch.float32),
                }
            )
            metric_rows.append(row)
    cross = compute_cross_shard_variant_consistency(
        metric_rows,
        threshold=float(config.get("thresholds", {}).get("cross_shard_consistency_min", 0.70)),
    )
    variant_names = sorted({row["variant_name"] for row in metric_rows})
    aggregates = {name: aggregate_variant_metrics(metric_rows, name) for name in variant_names}
    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in metric_rows]
    metrics = {
        "stage": STAGE,
        "num_rows": len(public_rows),
        "num_samples": len(samples),
        "variant_names": variant_names,
        "rows": public_rows,
        "aggregates": aggregates,
        "cross_shard_consistency": cross,
        "global_spatial_prior_stats": prior_payload["stats"],
        "training_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
    }
    ranking = rank_label_variants(metrics, config.get("thresholds", {}))
    comparison = {
        "stage": STAGE,
        "ranking": ranking,
        "best_variant": ranking.get("best_variant"),
        "num_variants": len(variant_names),
        "variant_names": variant_names,
        "score_is_engineering_heuristic": True,
        "redesigned_tensor_artifacts_saved": False,
    }
    summary = {
        "stage": STAGE,
        "safe_stop": False,
        "label_redesign_performed": True,
        "num_samples": len(samples),
        "num_samples_by_shard": _count_by(samples, "shard_id"),
        "num_variants": len(variant_names),
        "variant_names": variant_names,
        "global_spatial_prior_stats": prior_payload["stats"],
        "best_variant": ranking.get("best_variant"),
        "training_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "new_dataset_download_performed": False,
        "new_model_download_performed": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "future_tokens_used_as_input": False,
        "tensorflow_required": False,
        "tensorflow_datasets_required": False,
    }
    return {"summary": summary, "metrics": metrics, "comparison": comparison}


def write_step39a_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _label(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("label must be a torch.Tensor")
    tensor = value.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if tensor.ndim != 2:
        raise ValueError(f"label must be [T,S], got {tuple(tensor.shape)}")
    if bool(tensor.requires_grad):
        raise ValueError("label must not require gradients")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("label contains non-finite values")
    return tensor


def _prior(value: torch.Tensor, spatial_tokens: int) -> torch.Tensor:
    prior = value.detach().to(dtype=torch.float32, device="cpu").reshape(-1).contiguous()
    if int(prior.numel()) != int(spatial_tokens):
        raise ValueError("prior must have one value per spatial token")
    return prior


def _unit_minmax(value: torch.Tensor) -> torch.Tensor:
    tensor = value.detach().to(dtype=torch.float32, device="cpu")
    low, high = tensor.min(), tensor.max()
    if float((high - low).abs().item()) <= EPS:
        return torch.zeros_like(tensor)
    return ((tensor - low) / (high - low + EPS)).clamp(0.0, 1.0)


def _topk_mass_1d(value: torch.Tensor, k: int) -> float:
    flat = value.detach().to(dtype=torch.float32, device="cpu").reshape(-1).clamp_min(0.0)
    k = max(1, min(int(k), int(flat.numel())))
    return float(torch.topk(flat, k=k).values.sum().item() / (float(flat.sum().item()) + EPS))


def _mean(values: Any) -> float:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(sum(clean) / len(clean)) if clean else 0.0


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row[key])] = counts.get(str(row[key]), 0) + 1
    return counts


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
