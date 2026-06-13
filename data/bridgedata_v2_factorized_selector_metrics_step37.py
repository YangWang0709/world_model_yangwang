"""Metrics and gates for Step37 factorized selector diagnosis."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


STAGE = "bridgedata_v2_tfds_factorized_selector_step37"


def compute_factorized_prediction_metrics(
    pred_scores: torch.Tensor,
    target: torch.Tensor,
    *,
    random_seed: int = 42,
    topk_values: list[int] | None = None,
    step36_reference_mse: float | None = None,
) -> dict[str, Any]:
    topks = [int(k) for k in (topk_values or [64, 128, 256, 512])]
    pred = pred_scores.detach().to(dtype=torch.float32, device="cpu")
    tgt = target.detach().to(dtype=torch.float32, device="cpu")
    if pred.shape != tgt.shape or pred.ndim != 3:
        raise ValueError(f"pred and target must both be [B, T, S], got {tuple(pred.shape)} and {tuple(tgt.shape)}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    random_scores = torch.rand(tgt.shape, generator=generator, dtype=tgt.dtype)
    uniform = torch.full_like(tgt, 1.0 / float(tgt.shape[1] * tgt.shape[2]))
    temporal_broadcast = tgt.mean(dim=2, keepdim=True).expand_as(tgt)
    current_only = tgt.mean(dim=(1, 2), keepdim=True).expand_as(tgt)
    factorized_mse = _mse(pred, tgt)
    random_mse = _mse(random_scores, tgt)
    uniform_mse = _mse(uniform, tgt)
    temporal_mse = _mse(temporal_broadcast, tgt)
    current_mse = _mse(current_only, tgt)
    has_step36 = step36_reference_mse is not None and math.isfinite(float(step36_reference_mse))
    metrics: dict[str, Any] = {
        "factorized_selector_val_mse": factorized_mse,
        "random_token_baseline_mse": random_mse,
        "uniform_token_baseline_mse": uniform_mse,
        "temporal_broadcast_baseline_mse": temporal_mse,
        "current_only_patch_baseline_mse": current_mse,
        "step36_direct_patch_selector_mse_if_available": float(step36_reference_mse) if has_step36 else None,
        "factorized_beats_random": factorized_mse < random_mse,
        "factorized_beats_uniform": factorized_mse < uniform_mse,
        "factorized_beats_temporal_broadcast": factorized_mse < temporal_mse,
        "factorized_beats_current_only": factorized_mse < current_mse,
        "factorized_beats_step36_direct_patch": bool(has_step36 and factorized_mse < float(step36_reference_mse)),
        "pearson": _mean_corr(pred.reshape(pred.shape[0], -1), tgt.reshape(tgt.shape[0], -1), rank=False),
        "spearman": _mean_corr(pred.reshape(pred.shape[0], -1), tgt.reshape(tgt.shape[0], -1), rank=True),
        "num_eval_samples": int(tgt.shape[0]),
        "num_context_frames": int(tgt.shape[1]),
        "num_spatial_tokens": int(tgt.shape[2]),
        "num_patch_tokens": int(tgt.shape[1] * tgt.shape[2]),
    }
    for k in topks:
        metrics[f"top{k}_overlap"] = _topk_overlap(pred, tgt, k)
    metrics["top256_precision"] = _topk_overlap(pred, tgt, 256)
    metrics["top256_recall"] = metrics["top256_precision"]
    metrics["val_mse"] = metrics["factorized_selector_val_mse"]
    metrics["val_pearson"] = metrics["pearson"]
    metrics["val_spearman"] = metrics["spearman"]
    return metrics


def aggregate_factorized_metrics(rows: list[dict[str, Any]], *, eval_type: str) -> dict[str, Any]:
    comparable = [row for row in rows if not row.get("skipped")]
    if not comparable:
        return {
            "eval_type": eval_type,
            "num_rows": 0,
            "skipped": True,
            "factorized_selector_val_mse": 0.0,
            "random_token_baseline_mse": 0.0,
            "uniform_token_baseline_mse": 0.0,
            "temporal_broadcast_baseline_mse": 0.0,
            "current_only_patch_baseline_mse": 0.0,
            "step36_direct_patch_selector_mse_if_available": None,
            "factorized_beats_random": False,
            "factorized_beats_uniform": False,
            "factorized_beats_temporal_broadcast": False,
            "factorized_beats_current_only": False,
            "factorized_beats_step36_direct_patch": False,
            "pearson": 0.0,
            "spearman": 0.0,
            "top64_overlap": 0.0,
            "top128_overlap": 0.0,
            "top256_overlap": 0.0,
            "top512_overlap": 0.0,
            "top256_precision": 0.0,
            "top256_recall": 0.0,
            "overfit_gap": 0.0,
            "safety_gate_pass": True,
        }
    step36_values = [
        row["step36_direct_patch_selector_mse_if_available"]
        for row in comparable
        if row.get("step36_direct_patch_selector_mse_if_available") is not None
    ]
    return {
        "eval_type": eval_type,
        "num_rows": len(comparable),
        "rows": comparable,
        "factorized_selector_val_mse": _mean([row["factorized_selector_val_mse"] for row in comparable]),
        "random_token_baseline_mse": _mean([row["random_token_baseline_mse"] for row in comparable]),
        "uniform_token_baseline_mse": _mean([row["uniform_token_baseline_mse"] for row in comparable]),
        "temporal_broadcast_baseline_mse": _mean([row["temporal_broadcast_baseline_mse"] for row in comparable]),
        "current_only_patch_baseline_mse": _mean([row["current_only_patch_baseline_mse"] for row in comparable]),
        "step36_direct_patch_selector_mse_if_available": _mean(step36_values) if step36_values else None,
        "factorized_beats_random": all(bool(row["factorized_beats_random"]) for row in comparable),
        "factorized_beats_uniform": all(bool(row["factorized_beats_uniform"]) for row in comparable),
        "factorized_beats_temporal_broadcast": all(
            bool(row["factorized_beats_temporal_broadcast"]) for row in comparable
        ),
        "factorized_beats_current_only": all(bool(row["factorized_beats_current_only"]) for row in comparable),
        "factorized_beats_step36_direct_patch": all(
            bool(row["factorized_beats_step36_direct_patch"]) for row in comparable
        ),
        "pearson": _mean([row["pearson"] for row in comparable]),
        "spearman": _mean([row["spearman"] for row in comparable]),
        "top64_overlap": _mean([row["top64_overlap"] for row in comparable]),
        "top128_overlap": _mean([row["top128_overlap"] for row in comparable]),
        "top256_overlap": _mean([row["top256_overlap"] for row in comparable]),
        "top512_overlap": _mean([row["top512_overlap"] for row in comparable]),
        "top256_precision": _mean([row["top256_precision"] for row in comparable]),
        "top256_recall": _mean([row["top256_recall"] for row in comparable]),
        "overfit_gap": _mean([row.get("overfit_gap", 0.0) for row in comparable]),
        "all_finite": all(_row_finite(row) for row in comparable),
        "safety_gate_pass": True,
    }


def build_step37_gate_decision(
    *,
    within: dict[str, Any],
    cross: dict[str, Any],
    mixed: dict[str, Any],
    leakage: dict[str, Any],
    dataset_bias_detected: bool,
    full_context_noise_acknowledged: bool,
    top256_overlap_mean_min: float,
) -> dict[str, Any]:
    beats_random = all(bool(payload.get("factorized_beats_random")) for payload in (within, cross, mixed))
    beats_uniform = all(bool(payload.get("factorized_beats_uniform")) for payload in (within, cross, mixed))
    beats_temporal = all(bool(payload.get("factorized_beats_temporal_broadcast")) for payload in (within, cross, mixed))
    beats_current = all(bool(payload.get("factorized_beats_current_only")) for payload in (within, cross, mixed))
    beats_step36 = all(bool(payload.get("factorized_beats_step36_direct_patch")) for payload in (within, cross, mixed))
    top256_overlap_mean = _mean(
        [float(payload.get("top256_overlap", 0.0)) for payload in (within, cross, mixed) if int(payload.get("num_rows", 0)) > 0]
    )
    top256_pass = top256_overlap_mean >= float(top256_overlap_mean_min)
    cross_generalizes = (
        int(cross.get("num_rows", 0)) > 0
        and bool(cross.get("factorized_beats_temporal_broadcast"))
        and bool(cross.get("factorized_beats_current_only"))
        and bool(cross.get("factorized_beats_step36_direct_patch"))
        and float(cross.get("top256_overlap", 0.0)) >= float(top256_overlap_mean_min)
    )
    mixed_pass = (
        int(mixed.get("num_rows", 0)) > 0
        and bool(mixed.get("factorized_beats_temporal_broadcast"))
        and bool(mixed.get("factorized_beats_current_only"))
        and bool(mixed.get("factorized_beats_step36_direct_patch"))
        and float(mixed.get("top256_overlap", 0.0)) >= float(top256_overlap_mean_min)
    )
    no_leak = bool(leakage.get("no_language_or_trajectory_leakage", False))
    sample_ok = bool(leakage.get("train_val_sample_id_disjoint", False))
    traj_ok = bool(leakage.get("train_val_trajectory_disjoint_or_documented", False))
    future_ready = (
        beats_random
        and beats_uniform
        and beats_temporal
        and beats_current
        and beats_step36
        and cross_generalizes
        and top256_pass
        and no_leak
        and not bool(dataset_bias_detected)
        and bool(full_context_noise_acknowledged)
        and sample_ok
        and traj_ok
    )
    if future_ready:
        recommended = {
            "name": "bounded downstream utility smoke using factorized selector outputs",
            "scope": "do not train final selector; do not train current importance; keep context utility unclaimed",
        }
        reason = "factorized selector passed all bounded future-readiness diagnostics"
    else:
        recommended = {
            "name": "increase data or redesign proxy labels before downstream selector use",
            "scope": "keep selector/current-importance/final training disabled",
        }
        reason = "factorized selector did not pass one or more bounded diagnosis gates"
    return {
        "stage": STAGE,
        "future_factorized_selector_gate_ready": bool(future_ready),
        "factorized_selector_beats_random_token_baseline": bool(beats_random),
        "factorized_selector_beats_uniform_token_baseline": bool(beats_uniform),
        "factorized_selector_beats_temporal_broadcast_baseline": bool(beats_temporal),
        "factorized_selector_beats_current_only_patch_baseline": bool(beats_current),
        "factorized_selector_beats_step36_direct_patch_selector": bool(beats_step36),
        "factorized_selector_generalizes_cross_shard": bool(cross_generalizes),
        "mixed_package_pass": bool(mixed_pass),
        "top256_overlap_mean": float(top256_overlap_mean),
        "top256_overlap_above_threshold": bool(top256_pass),
        "top256_overlap_mean_min": float(top256_overlap_mean_min),
        "no_language_or_trajectory_leakage": bool(no_leak),
        "dataset_bias_detected": bool(dataset_bias_detected),
        "full_context_noise_acknowledged": bool(full_context_noise_acknowledged),
        "train_val_sample_id_disjoint": bool(sample_ok),
        "train_val_trajectory_disjoint_or_documented": bool(traj_ok),
        "selector_training_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "reason": reason,
        "recommended_step38": recommended,
        "safety_gate_pass": True,
    }


def _mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.mse_loss(pred, target, reduction="mean").item())


def _topk_overlap(pred: torch.Tensor, target: torch.Tensor, k: int) -> float:
    flat_pred = pred.reshape(pred.shape[0], -1)
    flat_target = target.reshape(target.shape[0], -1)
    kk = max(1, min(int(k), int(flat_target.shape[1])))
    pred_top = torch.topk(flat_pred, k=kk, dim=1).indices
    target_top = torch.topk(flat_target, k=kk, dim=1).indices
    hits = []
    for row in range(flat_pred.shape[0]):
        hits.append(len(set(pred_top[row].tolist()) & set(target_top[row].tolist())) / float(kk))
    return _mean(hits)


def _mean_corr(pred: torch.Tensor, target: torch.Tensor, *, rank: bool) -> float:
    values = []
    for left, right in zip(pred, target):
        a = _ranks(left) if rank else left
        b = _ranks(right) if rank else right
        values.append(_pearson(a, b))
    return _mean(values)


def _ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.to(dtype=torch.float32) - left.to(dtype=torch.float32).mean()
    b = right.to(dtype=torch.float32) - right.to(dtype=torch.float32).mean()
    denom = torch.sqrt(torch.sum(a * a) * torch.sum(b * b))
    if float(denom.item()) <= 1.0e-12:
        return 0.0
    return float((torch.sum(a * b) / denom).item())


def _row_finite(row: dict[str, Any]) -> bool:
    for value in row.values():
        if isinstance(value, float) and not math.isfinite(value):
            return False
    return True


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else 0.0
