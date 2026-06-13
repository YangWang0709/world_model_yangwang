"""Metrics and losses for Step36 proxy patch/token selector smoke training."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


STAGE = "bridgedata_v2_tfds_proxy_patch_selector_train_step36"


def combined_patch_selector_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if logits.shape != target.shape or logits.ndim != 3:
        raise ValueError(f"logits and target must both be [B, T, S], got {tuple(logits.shape)} and {tuple(target.shape)}")
    pred = torch.sigmoid(logits)
    mse = F.mse_loss(pred, target, reduction="mean")
    rank = sampled_patch_rank_loss(logits, target, num_pairs=int(config.get("rank_pairs_per_batch", 2048)))
    topk = patch_topk_soft_loss(logits, target, [int(k) for k in config.get("topk_values", [64, 128, 256, 512])])
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


def sampled_patch_rank_loss(logits: torch.Tensor, target: torch.Tensor, *, num_pairs: int) -> torch.Tensor:
    if logits.shape != target.shape or logits.ndim != 3:
        raise ValueError("logits and target must both be [B, T, S]")
    batch, _, _ = logits.shape
    flat_logits = logits.reshape(batch, -1)
    flat_target = target.reshape(batch, -1)
    num_tokens = int(flat_logits.shape[1])
    pairs = max(1, int(num_pairs))
    left = torch.randint(0, num_tokens, (batch, pairs), device=logits.device)
    right = torch.randint(0, num_tokens, (batch, pairs), device=logits.device)
    left_score = flat_logits.gather(1, left)
    right_score = flat_logits.gather(1, right)
    left_target = flat_target.gather(1, left)
    right_target = flat_target.gather(1, right)
    target_diff = left_target - right_target
    mask = torch.abs(target_diff) > 1.0e-6
    if not bool(mask.any()):
        return logits.new_tensor(0.0)
    sign = torch.sign(target_diff)
    score_diff = left_score - right_score
    return F.softplus(-score_diff[mask] * sign[mask]).mean()


def patch_topk_soft_loss(logits: torch.Tensor, target: torch.Tensor, topk_values: list[int]) -> torch.Tensor:
    if logits.shape != target.shape or logits.ndim != 3:
        raise ValueError("logits and target must both be [B, T, S]")
    flat_logits = logits.reshape(logits.shape[0], -1)
    flat_target = target.reshape(target.shape[0], -1)
    losses = []
    for k in topk_values:
        kk = max(1, min(int(k), int(flat_target.shape[1])))
        soft = torch.zeros_like(flat_target)
        indices = torch.topk(flat_target, k=kk, dim=1).indices
        soft.scatter_(1, indices, 1.0)
        losses.append(F.binary_cross_entropy_with_logits(flat_logits, soft, reduction="mean"))
    return torch.stack(losses).mean() if losses else logits.new_tensor(0.0)


def compute_patch_prediction_metrics(
    pred_scores: torch.Tensor,
    target: torch.Tensor,
    *,
    random_seed: int = 42,
    topk_values: list[int] | None = None,
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
    patch_mse = _mse(pred, tgt)
    random_mse = _mse(random_scores, tgt)
    uniform_mse = _mse(uniform, tgt)
    temporal_mse = _mse(temporal_broadcast, tgt)
    current_mse = _mse(current_only, tgt)
    metrics: dict[str, Any] = {
        "patch_selector_val_mse": patch_mse,
        "random_token_baseline_mse": random_mse,
        "uniform_token_baseline_mse": uniform_mse,
        "temporal_broadcast_baseline_mse": temporal_mse,
        "current_only_patch_baseline_mse": current_mse,
        "patch_selector_beats_random_token_baseline": patch_mse < random_mse,
        "patch_selector_beats_uniform_token_baseline": patch_mse < uniform_mse,
        "patch_selector_beats_temporal_broadcast_baseline": patch_mse < temporal_mse,
        "patch_selector_beats_current_only_patch_baseline": patch_mse < current_mse,
        "patch_selector_spearman": _mean_corr(pred.reshape(pred.shape[0], -1), tgt.reshape(tgt.shape[0], -1), rank=True),
        "patch_selector_pearson": _mean_corr(pred.reshape(pred.shape[0], -1), tgt.reshape(tgt.shape[0], -1), rank=False),
        "num_eval_samples": int(tgt.shape[0]),
        "num_context_frames": int(tgt.shape[1]),
        "num_spatial_tokens": int(tgt.shape[2]),
        "num_patch_tokens": int(tgt.shape[1] * tgt.shape[2]),
    }
    for k in topks:
        metrics[f"top{k}_overlap"] = _topk_overlap(pred, tgt, k)
    metrics["top256_precision"] = _topk_overlap(pred, tgt, 256)
    metrics["top256_recall"] = metrics["top256_precision"]
    metrics["val_mse"] = metrics["patch_selector_val_mse"]
    metrics["val_spearman"] = metrics["patch_selector_spearman"]
    metrics["val_pearson"] = metrics["patch_selector_pearson"]
    return metrics


def aggregate_setting_metrics(rows: list[dict[str, Any]], *, eval_type: str) -> dict[str, Any]:
    comparable = [row for row in rows if not row.get("skipped")]
    if not comparable:
        return {
            "eval_type": eval_type,
            "num_rows": 0,
            "skipped": True,
            "patch_selector_val_mse": 0.0,
            "random_token_baseline_mse": 0.0,
            "uniform_token_baseline_mse": 0.0,
            "temporal_broadcast_baseline_mse": 0.0,
            "current_only_patch_baseline_mse": 0.0,
            "patch_selector_beats_random_token_baseline": False,
            "patch_selector_beats_uniform_token_baseline": False,
            "patch_selector_beats_temporal_broadcast_baseline": False,
            "patch_selector_beats_current_only_patch_baseline": False,
            "patch_selector_spearman": 0.0,
            "patch_selector_pearson": 0.0,
            "top64_overlap": 0.0,
            "top128_overlap": 0.0,
            "top256_overlap": 0.0,
            "top512_overlap": 0.0,
            "top256_precision": 0.0,
            "top256_recall": 0.0,
            "overfit_gap": 0.0,
            "safety_gate_pass": True,
        }
    return {
        "eval_type": eval_type,
        "num_rows": len(comparable),
        "rows": comparable,
        "patch_selector_val_mse": _mean([row["patch_selector_val_mse"] for row in comparable]),
        "random_token_baseline_mse": _mean([row["random_token_baseline_mse"] for row in comparable]),
        "uniform_token_baseline_mse": _mean([row["uniform_token_baseline_mse"] for row in comparable]),
        "temporal_broadcast_baseline_mse": _mean([row["temporal_broadcast_baseline_mse"] for row in comparable]),
        "current_only_patch_baseline_mse": _mean([row["current_only_patch_baseline_mse"] for row in comparable]),
        "patch_selector_spearman": _mean([row["patch_selector_spearman"] for row in comparable]),
        "patch_selector_pearson": _mean([row["patch_selector_pearson"] for row in comparable]),
        "top64_overlap": _mean([row["top64_overlap"] for row in comparable]),
        "top128_overlap": _mean([row["top128_overlap"] for row in comparable]),
        "top256_overlap": _mean([row["top256_overlap"] for row in comparable]),
        "top512_overlap": _mean([row["top512_overlap"] for row in comparable]),
        "top256_precision": _mean([row["top256_precision"] for row in comparable]),
        "top256_recall": _mean([row["top256_recall"] for row in comparable]),
        "patch_selector_beats_random_token_baseline": all(
            bool(row["patch_selector_beats_random_token_baseline"]) for row in comparable
        ),
        "patch_selector_beats_uniform_token_baseline": all(
            bool(row["patch_selector_beats_uniform_token_baseline"]) for row in comparable
        ),
        "patch_selector_beats_temporal_broadcast_baseline": all(
            bool(row["patch_selector_beats_temporal_broadcast_baseline"]) for row in comparable
        ),
        "patch_selector_beats_current_only_patch_baseline": all(
            bool(row["patch_selector_beats_current_only_patch_baseline"]) for row in comparable
        ),
        "overfit_gap": _mean([row.get("overfit_gap", 0.0) for row in comparable]),
        "all_finite": all(_row_finite(row) for row in comparable),
        "safety_gate_pass": True,
    }


def build_step36_gate_decision(
    *,
    within: dict[str, Any],
    cross: dict[str, Any],
    mixed: dict[str, Any],
    leakage: dict[str, Any],
    dataset_bias_detected: bool,
    full_context_noise_acknowledged: bool,
    top256_overlap_mean_min: float = 0.10,
) -> dict[str, Any]:
    beats_random = all(
        bool(payload.get("patch_selector_beats_random_token_baseline")) for payload in (within, cross, mixed)
    )
    beats_uniform = all(
        bool(payload.get("patch_selector_beats_uniform_token_baseline")) for payload in (within, cross, mixed)
    )
    beats_temporal = all(
        bool(payload.get("patch_selector_beats_temporal_broadcast_baseline")) for payload in (within, cross, mixed)
    )
    beats_current = all(
        bool(payload.get("patch_selector_beats_current_only_patch_baseline")) for payload in (within, cross, mixed)
    )
    top256_overlap_mean = _mean(
        [float(payload.get("top256_overlap", 0.0)) for payload in (within, cross, mixed) if int(payload.get("num_rows", 0)) > 0]
    )
    top256_pass = top256_overlap_mean >= float(top256_overlap_mean_min)
    cross_generalizes = (
        int(cross.get("num_rows", 0)) > 0
        and bool(cross.get("patch_selector_beats_random_token_baseline"))
        and bool(cross.get("patch_selector_beats_uniform_token_baseline"))
        and bool(cross.get("patch_selector_beats_temporal_broadcast_baseline"))
        and float(cross.get("top256_overlap", 0.0)) >= float(top256_overlap_mean_min)
    )
    mixed_pass = (
        int(mixed.get("num_rows", 0)) > 0
        and bool(mixed.get("patch_selector_beats_random_token_baseline"))
        and bool(mixed.get("patch_selector_beats_uniform_token_baseline"))
        and bool(mixed.get("patch_selector_beats_temporal_broadcast_baseline"))
        and float(mixed.get("top256_overlap", 0.0)) >= float(top256_overlap_mean_min)
    )
    no_leak = bool(leakage.get("no_language_or_trajectory_leakage", False))
    sample_ok = bool(leakage.get("train_val_sample_id_disjoint", False))
    traj_ok = bool(leakage.get("train_val_trajectory_disjoint_or_documented", False))
    future_ready = (
        beats_random
        and beats_uniform
        and beats_temporal
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
            "name": "bounded downstream utility smoke using learned patch selector outputs",
            "scope": "do not train final selector yet; do not train current importance yet; keep context utility unclaimed",
        }
        reason = "bounded patch/token selector smoke passed future-readiness gates"
    else:
        recommended = {
            "name": "improve patch selector architecture or increase data before downstream selector use",
            "scope": "keep final selector and current-importance training disabled",
        }
        reason = "one or more bounded patch/token smoke gates did not pass"
    return {
        "stage": STAGE,
        "future_patch_selector_gate_ready": bool(future_ready),
        "patch_selector_beats_random_token_baseline": bool(beats_random),
        "patch_selector_beats_uniform_token_baseline": bool(beats_uniform),
        "patch_selector_beats_temporal_broadcast_baseline": bool(beats_temporal),
        "patch_selector_beats_current_only_patch_baseline": bool(beats_current),
        "patch_selector_generalizes_cross_shard": bool(cross_generalizes),
        "mixed_shard_pass": bool(mixed_pass),
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
        "recommended_step37": recommended,
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
