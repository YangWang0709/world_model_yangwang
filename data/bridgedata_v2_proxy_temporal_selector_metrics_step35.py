"""Metrics and losses for Step35 proxy-temporal selector smoke training."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


STAGE = "bridgedata_v2_tfds_proxy_temporal_selector_train_step35"


def combined_temporal_selector_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    pred = torch.sigmoid(logits)
    mse = F.mse_loss(pred, target, reduction="mean")
    rank = temporal_rank_loss(logits, target)
    topk = temporal_topk_soft_loss(logits, target, [int(k) for k in config.get("topk_frames", [1, 2, 4])])
    total = (
        float(config.get("mse_weight", 1.0)) * mse
        + float(config.get("rank_weight", 0.1)) * rank
        + float(config.get("topk_soft_weight", 0.1)) * topk
    )
    return {
        "loss": total,
        "mse_loss": mse.detach(),
        "rank_loss": rank.detach(),
        "topk_soft_loss": topk.detach(),
    }


def temporal_rank_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError("logits and target must have the same shape")
    score_diff = logits.unsqueeze(2) - logits.unsqueeze(1)
    target_diff = target.unsqueeze(2) - target.unsqueeze(1)
    sign = torch.sign(target_diff)
    mask = torch.abs(target_diff) > 1.0e-6
    if not bool(mask.any()):
        return logits.new_tensor(0.0)
    return F.softplus(-score_diff[mask] * sign[mask]).mean()


def temporal_topk_soft_loss(logits: torch.Tensor, target: torch.Tensor, topk_frames: list[int]) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError("logits and target must have the same shape")
    losses = []
    for k in topk_frames:
        kk = max(1, min(int(k), int(target.shape[1])))
        soft = torch.zeros_like(target)
        indices = torch.topk(target, k=kk, dim=1).indices
        soft.scatter_(1, indices, 1.0)
        losses.append(F.binary_cross_entropy_with_logits(logits, soft, reduction="mean"))
    return torch.stack(losses).mean() if losses else logits.new_tensor(0.0)


def compute_temporal_prediction_metrics(
    pred_scores: torch.Tensor,
    target: torch.Tensor,
    *,
    random_seed: int = 42,
) -> dict[str, Any]:
    pred = pred_scores.detach().to(dtype=torch.float32, device="cpu")
    tgt = target.detach().to(dtype=torch.float32, device="cpu")
    if pred.shape != tgt.shape or pred.ndim != 2:
        raise ValueError(f"pred and target must both be [B, T], got {tuple(pred.shape)} and {tuple(tgt.shape)}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    random_scores = torch.rand(tgt.shape, generator=generator, dtype=tgt.dtype)
    current_only = tgt.mean(dim=1, keepdim=True).expand_as(tgt)
    uniform = torch.full_like(tgt, 1.0 / float(tgt.shape[1]))
    selector_mse = _mse(pred, tgt)
    random_mse = _mse(random_scores, tgt)
    current_mse = _mse(current_only, tgt)
    uniform_mse = _mse(uniform, tgt)
    metrics = {
        "selector_val_mse": selector_mse,
        "random_baseline_mse": random_mse,
        "current_only_baseline_mse": current_mse,
        "uniform_baseline_mse": uniform_mse,
        "selector_beats_random_baseline": selector_mse < random_mse,
        "selector_beats_current_only_baseline": selector_mse < current_mse,
        "selector_beats_uniform_baseline": selector_mse < uniform_mse,
        "selector_spearman": _mean_corr(pred, tgt, rank=True),
        "selector_pearson": _mean_corr(pred, tgt, rank=False),
        "top1_frame_hit": _topk_overlap(pred, tgt, 1),
        "top2_frame_overlap": _topk_overlap(pred, tgt, 2),
        "top4_frame_overlap": _topk_overlap(pred, tgt, 4),
        "num_eval_samples": int(tgt.shape[0]),
        "num_frames": int(tgt.shape[1]),
    }
    metrics["val_mse"] = metrics["selector_val_mse"]
    metrics["val_spearman"] = metrics["selector_spearman"]
    metrics["val_pearson"] = metrics["selector_pearson"]
    return metrics


def aggregate_setting_metrics(rows: list[dict[str, Any]], *, eval_type: str) -> dict[str, Any]:
    comparable = [row for row in rows if not row.get("skipped")]
    if not comparable:
        return {
            "eval_type": eval_type,
            "num_rows": 0,
            "skipped": True,
            "selector_beats_random_baseline": False,
            "selector_beats_current_only_baseline": False,
            "selector_beats_uniform_baseline": False,
            "selector_val_mse": 0.0,
            "random_baseline_mse": 0.0,
            "current_only_baseline_mse": 0.0,
            "uniform_baseline_mse": 0.0,
            "selector_spearman": 0.0,
            "selector_pearson": 0.0,
            "top1_frame_hit": 0.0,
            "top2_frame_overlap": 0.0,
            "top4_frame_overlap": 0.0,
            "overfit_gap": 0.0,
            "safety_gate_pass": True,
        }
    return {
        "eval_type": eval_type,
        "num_rows": len(comparable),
        "rows": comparable,
        "selector_val_mse": _mean([row["selector_val_mse"] for row in comparable]),
        "random_baseline_mse": _mean([row["random_baseline_mse"] for row in comparable]),
        "current_only_baseline_mse": _mean([row["current_only_baseline_mse"] for row in comparable]),
        "uniform_baseline_mse": _mean([row["uniform_baseline_mse"] for row in comparable]),
        "selector_spearman": _mean([row["selector_spearman"] for row in comparable]),
        "selector_pearson": _mean([row["selector_pearson"] for row in comparable]),
        "top1_frame_hit": _mean([row["top1_frame_hit"] for row in comparable]),
        "top2_frame_overlap": _mean([row["top2_frame_overlap"] for row in comparable]),
        "top4_frame_overlap": _mean([row["top4_frame_overlap"] for row in comparable]),
        "selector_beats_random_baseline": all(bool(row["selector_beats_random_baseline"]) for row in comparable),
        "selector_beats_current_only_baseline": all(
            bool(row["selector_beats_current_only_baseline"]) for row in comparable
        ),
        "selector_beats_uniform_baseline": all(bool(row["selector_beats_uniform_baseline"]) for row in comparable),
        "overfit_gap": _mean([row.get("overfit_gap", 0.0) for row in comparable]),
        "all_finite": all(_row_finite(row) for row in comparable),
        "safety_gate_pass": True,
    }


def build_step35_gate_decision(
    *,
    within: dict[str, Any],
    cross: dict[str, Any],
    mixed: dict[str, Any],
    leakage: dict[str, Any],
    dataset_bias_detected: bool,
    full_context_noise_acknowledged: bool,
) -> dict[str, Any]:
    selector_beats_random = bool(within.get("selector_beats_random_baseline")) and bool(
        cross.get("selector_beats_random_baseline")
    ) and bool(mixed.get("selector_beats_random_baseline"))
    selector_beats_current = bool(within.get("selector_beats_current_only_baseline")) and bool(
        cross.get("selector_beats_current_only_baseline")
    ) and bool(mixed.get("selector_beats_current_only_baseline"))
    cross_generalizes = (
        int(cross.get("num_rows", 0)) > 0
        and bool(cross.get("selector_beats_random_baseline"))
        and bool(cross.get("selector_beats_current_only_baseline"))
    )
    mixed_pass = (
        int(mixed.get("num_rows", 0)) > 0
        and bool(mixed.get("selector_beats_random_baseline"))
        and bool(mixed.get("selector_beats_current_only_baseline"))
    )
    no_leak = bool(leakage.get("no_language_or_trajectory_leakage", False))
    sample_ok = bool(leakage.get("train_val_sample_id_disjoint", False))
    traj_ok = bool(leakage.get("train_val_trajectory_disjoint_or_documented", False))
    future_ready = (
        selector_beats_random
        and selector_beats_current
        and cross_generalizes
        and no_leak
        and not bool(dataset_bias_detected)
        and bool(full_context_noise_acknowledged)
        and sample_ok
        and traj_ok
    )
    if future_ready:
        recommended = {
            "name": "bounded patch-level selector planning or temporal-selector downstream utility smoke",
            "scope": "do not train final selector yet; do not train current importance yet",
        }
        reason = "bounded temporal selector smoke passed future-readiness gates"
    else:
        recommended = {
            "name": "improve selector architecture or increase data before selector training",
            "scope": "keep final selector and current-importance training disabled",
        }
        reason = "one or more bounded smoke gates did not pass"
    return {
        "stage": STAGE,
        "future_selector_training_gate_ready": bool(future_ready),
        "selector_beats_random_baseline": bool(selector_beats_random),
        "selector_beats_current_only_baseline": bool(selector_beats_current),
        "selector_generalizes_cross_shard": bool(cross_generalizes),
        "mixed_shard_pass": bool(mixed_pass),
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
        "recommended_step36": recommended,
        "safety_gate_pass": True,
    }


def _mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.mse_loss(pred, target, reduction="mean").item())


def _topk_overlap(pred: torch.Tensor, target: torch.Tensor, k: int) -> float:
    kk = max(1, min(int(k), int(target.shape[1])))
    pred_top = torch.topk(pred, k=kk, dim=1).indices
    target_top = torch.topk(target, k=kk, dim=1).indices
    hits = []
    for row in range(pred.shape[0]):
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
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            return False
    return True


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else 0.0
