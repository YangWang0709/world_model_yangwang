"""Metrics and decision helpers for Step29 train/val context utility sanity."""

from __future__ import annotations

import math
from typing import Any


ALLOWED_CLAIM = "train-val sanity only; not final paper result"


def summarize_train_val_curve(
    policy: str,
    curve: list[dict[str, Any]],
    min_relative_train_loss_decrease: float = 0.01,
) -> dict[str, Any]:
    train_losses = [float(point["train_loss"]) for point in curve]
    val_losses = [float(point["val_loss"]) for point in curve]
    train_initial = train_losses[0] if train_losses else math.nan
    train_final = train_losses[-1] if train_losses else math.nan
    val_initial = val_losses[0] if val_losses else math.nan
    val_final = val_losses[-1] if val_losses else math.nan
    train_best = min([value for value in train_losses if math.isfinite(value)], default=math.nan)
    val_best = min([value for value in val_losses if math.isfinite(value)], default=math.nan)
    train_relative = _relative_decrease(train_initial, train_final)
    val_relative = _relative_decrease(val_initial, val_final)
    return {
        "policy": policy,
        "train_initial_loss": float(train_initial),
        "train_final_loss": float(train_final),
        "train_best_loss": float(train_best),
        "val_initial_loss": float(val_initial),
        "val_final_loss": float(val_final),
        "val_best_loss": float(val_best),
        "train_relative_loss_decrease": float(train_relative),
        "val_relative_loss_decrease": float(val_relative),
        "train_loss_decreased": bool(
            math.isfinite(train_relative) and train_relative >= float(min_relative_train_loss_decrease)
        ),
        "val_loss_finite": bool(val_losses and all(math.isfinite(value) for value in val_losses)),
        "num_curve_points": len(curve),
    }


def build_context_utility_comparison(
    policy_metrics: list[dict[str, Any]],
    positive_requires: dict[str, float] | None = None,
) -> dict[str, Any]:
    positive_requires = positive_requires or {}
    by_policy = {str(item["policy"]): item for item in policy_metrics}
    current = by_policy.get("current_only", {})
    random = by_policy.get("random_context_topk", {})
    proxy = by_policy.get("proxy_importance_topk", {})
    full = by_policy.get("full_context_reference", {})
    current_loss = _loss(current)
    random_loss = _loss(random)
    proxy_loss = _loss(proxy)
    full_loss = _loss(full)

    proxy_vs_current = _relative_improvement(current_loss, proxy_loss)
    proxy_vs_random = _relative_improvement(random_loss, proxy_loss)
    full_vs_current = _relative_improvement(current_loss, full_loss)
    proxy_current_threshold = float(positive_requires.get("proxy_better_than_current_only_relative", 0.01))
    proxy_random_threshold = float(positive_requires.get("proxy_better_than_random_relative", 0.01))
    full_current_threshold = float(positive_requires.get("full_better_than_current_only_relative", 0.01))
    proxy_better_current = proxy_vs_current >= proxy_current_threshold
    proxy_better_random = proxy_vs_random >= proxy_random_threshold
    full_better_current = full_vs_current >= full_current_threshold
    current_best = current_loss <= min(random_loss, proxy_loss, full_loss)
    if proxy_better_current and proxy_better_random and full_better_current:
        signal = "positive"
        reason = "proxy context and full context improve held-out loss over current-only under the sanity threshold."
        recommended = "trained-predictor occlusion teacher on expanded train split"
    elif full_better_current and not (proxy_better_current and proxy_better_random):
        signal = "inconclusive_teacher_or_selector_issue"
        reason = "full context helps, but proxy top-k does not clearly beat current-only and random top-k."
        recommended = "improve teacher label or increase window diversity before selector training"
    elif current_best:
        signal = "negative_or_current_dominant"
        reason = "current_only is best on held-out validation, so context utility is not supported by this sanity run."
        recommended = "diagnose horizon/current dominance and reconsider context objective before selector training"
    else:
        signal = "inconclusive"
        reason = "held-out policy ordering is mixed and does not justify a context utility claim."
        recommended = "improve teacher label or increase window diversity before selector training"
    return {
        "current_only_val_final_loss": current_loss,
        "random_context_topk_val_final_loss": random_loss,
        "proxy_importance_topk_val_final_loss": proxy_loss,
        "full_context_reference_val_final_loss": full_loss,
        "proxy_better_than_current_only": bool(proxy_better_current),
        "proxy_better_than_random": bool(proxy_better_random),
        "full_better_than_current_only": bool(full_better_current),
        "proxy_val_relative_improvement_over_current_only": float(proxy_vs_current),
        "proxy_val_relative_improvement_over_random": float(proxy_vs_random),
        "full_val_relative_improvement_over_current_only": float(full_vs_current),
        "all_val_losses_finite": all(
            math.isfinite(value) for value in (current_loss, random_loss, proxy_loss, full_loss)
        ),
        "context_utility_signal": signal,
        "context_utility_sanity_signal": signal,
        "context_utility_claim_allowed": False,
        "allowed_claim": ALLOWED_CLAIM,
        "reason": reason,
        "recommended_step30": recommended,
        "train_current_importance": False,
    }


def _loss(metric: dict[str, Any]) -> float:
    return float(metric.get("val_final_loss", math.inf))


def _relative_decrease(initial: float, final: float) -> float:
    if not math.isfinite(initial) or abs(initial) <= 0.0:
        return 0.0
    return float((initial - final) / abs(initial))


def _relative_improvement(baseline_loss: float, candidate_loss: float) -> float:
    if not math.isfinite(baseline_loss) or abs(baseline_loss) <= 0.0:
        return 0.0
    return float((baseline_loss - candidate_loss) / abs(baseline_loss))

