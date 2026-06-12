"""Aggregate Step30A per-seed policy metrics."""

from __future__ import annotations

import math
from typing import Any


POLICIES = ("current_only", "random_context_topk", "proxy_importance_topk", "full_context_reference")


def aggregate_window_diversity_metrics(trainval_runs: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    runs = trainval_runs.get("runs", []) if isinstance(trainval_runs, dict) else trainval_runs
    policy_losses: dict[str, list[float]] = {policy: [] for policy in POLICIES}
    policy_best_losses: dict[str, list[float]] = {policy: [] for policy in POLICIES}
    for run in runs:
        by_policy = {str(item["policy"]): item for item in run.get("policy_metrics", [])}
        for policy in POLICIES:
            metric = by_policy.get(policy)
            if not metric:
                continue
            policy_losses[policy].append(float(metric.get("val_final_loss", math.inf)))
            policy_best_losses[policy].append(float(metric.get("val_best_loss", math.inf)))

    return {
        "num_runs": len(runs),
        "policy_val_final_mean": {policy: _mean(values) for policy, values in policy_losses.items()},
        "policy_val_best_mean": {policy: _mean(values) for policy, values in policy_best_losses.items()},
        "policy_val_final_values": policy_losses,
        "proxy_beats_current_count": _count_flag(runs, "proxy_better_than_current_only"),
        "proxy_beats_random_count": _count_flag(runs, "proxy_better_than_random"),
        "full_beats_current_count": _count_flag(runs, "full_better_than_current_only"),
        "all_val_losses_finite": all(
            math.isfinite(value) for values in policy_losses.values() for value in values
        ),
    }


def recommended_step30b_from_signal(signal: str) -> dict[str, str]:
    if signal == "positive_but_not_final":
        return {
            "name": "trained-predictor occlusion teacher on expanded windows",
            "condition": "proxy and full context both repeatedly improve held-out loss",
            "scope": "no selector/current-importance training yet",
        }
    if signal == "weak_proxy_positive_full_context_noisy":
        return {
            "name": "trained-predictor occlusion teacher or add more data diversity",
            "condition": "proxy top-k is stable but full context is noisy",
            "scope": "no selector/current-importance training yet",
        }
    if signal == "negative_or_current_dominant":
        return {
            "name": "diagnose horizon/current dominance before selector training",
            "condition": "context policies do not consistently beat current_only",
            "scope": "representation/horizon diagnosis only",
        }
    return {
        "name": "increase data diversity or improve temporal tokenization before teacher/selector",
        "condition": "window diversity sanity remains mixed",
        "scope": "no selector/current-importance training yet",
    }


def _count_flag(runs: list[dict[str, Any]], key: str) -> int:
    return sum(1 for run in runs if bool(run.get("context_comparison", {}).get(key)))


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))
