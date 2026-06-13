"""Metrics for Step40A redesigned-label selector smoke."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


STAGE = "bridgedata_v2_tfds_redesigned_label_selector_step40a"


def compute_redesigned_label_selector_metrics(
    pred_scores: torch.Tensor,
    target: torch.Tensor,
    train_prior: torch.Tensor,
    *,
    random_seed: int = 42,
    topk_values: list[int] | None = None,
) -> dict[str, Any]:
    topks = [int(k) for k in (topk_values or [64, 128, 256, 512])]
    pred = pred_scores.detach().to(dtype=torch.float32, device="cpu")
    tgt = target.detach().to(dtype=torch.float32, device="cpu")
    prior = train_prior.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    if pred.shape != tgt.shape or pred.ndim != 3:
        raise ValueError("pred and target must both be [B,T,S]")
    if int(prior.numel()) != int(tgt.shape[2]):
        raise ValueError("train prior must have one value per spatial token")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    random_scores = torch.rand(tgt.shape, generator=generator, dtype=tgt.dtype)
    mean_baseline = tgt.mean(dim=(1, 2), keepdim=True).expand_as(tgt)
    temporal = tgt.mean(dim=2, keepdim=True).expand_as(tgt)
    prior_baseline = prior.view(1, 1, -1).expand_as(tgt)
    selector_mse = _mse(pred, tgt)
    metrics: dict[str, Any] = {
        "selector_val_mse": selector_mse,
        "selector_val_mae": _mae(pred, tgt),
        "random_token_baseline_mse": _mse(random_scores, tgt),
        "uniform_or_mean_baseline_mse": _mse(mean_baseline, tgt),
        "temporal_broadcast_baseline_mse": _mse(temporal, tgt),
        "train_global_spatial_prior_baseline_mse": _mse(prior_baseline, tgt),
        "pearson": _mean_corr(pred.reshape(pred.shape[0], -1), tgt.reshape(tgt.shape[0], -1), rank=False),
        "spearman": _mean_corr(pred.reshape(pred.shape[0], -1), tgt.reshape(tgt.shape[0], -1), rank=True),
        "num_eval_samples": int(tgt.shape[0]),
        "num_context_frames": int(tgt.shape[1]),
        "num_spatial_tokens": int(tgt.shape[2]),
        "all_finite": bool(torch.isfinite(pred).all() and torch.isfinite(tgt).all()),
    }
    metrics["selector_beats_random_token_baseline"] = selector_mse < metrics["random_token_baseline_mse"]
    metrics["selector_beats_uniform_or_mean_baseline"] = selector_mse < metrics["uniform_or_mean_baseline_mse"]
    metrics["selector_beats_temporal_broadcast_baseline"] = selector_mse < metrics["temporal_broadcast_baseline_mse"]
    metrics["selector_beats_train_global_spatial_prior_baseline"] = (
        selector_mse < metrics["train_global_spatial_prior_baseline_mse"]
    )
    for k in topks:
        metrics[f"top{k}_overlap"] = _topk_overlap(pred, tgt, k)
    metrics["top256_precision"] = metrics.get("top256_overlap", _topk_overlap(pred, tgt, 256))
    metrics["top256_recall"] = metrics["top256_precision"]
    return metrics


def aggregate_redesigned_label_selector_metrics(rows: list[dict[str, Any]], *, eval_type: str) -> dict[str, Any]:
    comparable = [row for row in rows if not row.get("skipped")]
    if not comparable:
        return {
            "eval_type": eval_type,
            "num_rows": 0,
            "skipped": True,
            "selector_val_mse": 0.0,
            "selector_val_mae": 0.0,
            "random_token_baseline_mse": 0.0,
            "uniform_or_mean_baseline_mse": 0.0,
            "temporal_broadcast_baseline_mse": 0.0,
            "train_global_spatial_prior_baseline_mse": 0.0,
            "selector_beats_random_token_baseline": False,
            "selector_beats_uniform_or_mean_baseline": False,
            "selector_beats_temporal_broadcast_baseline": False,
            "selector_beats_train_global_spatial_prior_baseline": False,
            "pearson": 0.0,
            "spearman": 0.0,
            "top256_overlap": 0.0,
            "overfit_gap": 0.0,
            "all_finite": True,
        }
    keys = [
        "selector_val_mse",
        "selector_val_mae",
        "random_token_baseline_mse",
        "uniform_or_mean_baseline_mse",
        "temporal_broadcast_baseline_mse",
        "train_global_spatial_prior_baseline_mse",
        "pearson",
        "spearman",
        "top64_overlap",
        "top128_overlap",
        "top256_overlap",
        "top512_overlap",
        "top256_precision",
        "top256_recall",
        "train_mse",
        "overfit_gap",
    ]
    payload = {"eval_type": eval_type, "num_rows": len(comparable), "rows": comparable}
    payload.update({key: _mean([row.get(key, 0.0) for row in comparable]) for key in keys})
    for flag in (
        "selector_beats_random_token_baseline",
        "selector_beats_uniform_or_mean_baseline",
        "selector_beats_temporal_broadcast_baseline",
        "selector_beats_train_global_spatial_prior_baseline",
    ):
        payload[flag] = all(bool(row.get(flag, False)) for row in comparable)
    payload["all_finite"] = all(bool(row.get("all_finite", False)) and _row_finite(row) for row in comparable)
    payload["safety_gate_pass"] = True
    return payload


def build_step40a_gate_decision(
    *,
    within: dict[str, Any],
    cross: dict[str, Any],
    mixed: dict[str, Any],
    leakage: dict[str, Any],
    requirements: dict[str, Any],
) -> dict[str, Any]:
    payloads = (within, cross, mixed)
    beats_random = all(bool(payload.get("selector_beats_random_token_baseline", False)) for payload in payloads)
    beats_mean = all(bool(payload.get("selector_beats_uniform_or_mean_baseline", False)) for payload in payloads)
    beats_temporal = all(bool(payload.get("selector_beats_temporal_broadcast_baseline", False)) for payload in payloads)
    beats_prior = all(bool(payload.get("selector_beats_train_global_spatial_prior_baseline", False)) for payload in payloads)
    top256_mean = _mean(
        [float(payload.get("top256_overlap", 0.0)) for payload in payloads if int(payload.get("num_rows", 0)) > 0]
    )
    overfit_mean = _mean(
        [abs(float(payload.get("overfit_gap", 0.0))) for payload in payloads if int(payload.get("num_rows", 0)) > 0]
    )
    top256_pass = top256_mean >= float(requirements.get("top256_overlap_mean_min", 0.12))
    overfit_pass = overfit_mean <= float(requirements.get("overfit_gap_max", 0.05))
    cross_generalizes = (
        int(cross.get("num_rows", 0)) > 0
        and bool(cross.get("selector_beats_random_token_baseline", False))
        and bool(cross.get("selector_beats_uniform_or_mean_baseline", False))
        and bool(cross.get("selector_beats_temporal_broadcast_baseline", False))
        and bool(cross.get("selector_beats_train_global_spatial_prior_baseline", False))
        and float(cross.get("top256_overlap", 0.0)) >= float(requirements.get("top256_overlap_mean_min", 0.12))
    )
    no_leak = bool(leakage.get("no_language_or_trajectory_leakage", False))
    smoke_pass = bool(
        beats_random
        and beats_mean
        and beats_temporal
        and beats_prior
        and cross_generalizes
        and top256_pass
        and overfit_pass
        and no_leak
    )
    if smoke_pass:
        recommended = {
            "name": "bounded downstream utility smoke using redesigned-label selector outputs",
            "scope": "no final selector, no current importance, no downstream utility claim",
        }
        reason = "bounded redesigned-label selector smoke passed all configured gates"
    else:
        recommended = {
            "name": "current-conditioning diagnosis or encoder/label redesign",
            "scope": "no downstream selector use",
        }
        reason = "one or more redesigned-label selector smoke gates did not pass"
    return {
        "stage": STAGE,
        "redesigned_label_selector_smoke_pass": smoke_pass,
        "selector_beats_random_token_baseline": bool(beats_random),
        "selector_beats_uniform_or_mean_baseline": bool(beats_mean),
        "selector_beats_temporal_broadcast_baseline": bool(beats_temporal),
        "selector_beats_train_global_spatial_prior_baseline": bool(beats_prior),
        "selector_generalizes_cross_shard": bool(cross_generalizes),
        "top256_overlap_mean": float(top256_mean),
        "top256_overlap_above_threshold": bool(top256_pass),
        "top256_overlap_mean_min": float(requirements.get("top256_overlap_mean_min", 0.12)),
        "overfit_gap_mean_abs": float(overfit_mean),
        "overfit_gap_pass": bool(overfit_pass),
        "no_language_or_trajectory_leakage": bool(no_leak),
        "bounded_selector_smoke_performed": True,
        "downstream_selector_use_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "reason": reason,
        "recommended_step41": recommended,
        "safety_gate_pass": True,
    }


def _mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.mse_loss(pred, target, reduction="mean").item())


def _mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).item())


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
    return _mean([_pearson(_ranks(a) if rank else a, _ranks(b) if rank else b) for a, b in zip(pred, target)])


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
