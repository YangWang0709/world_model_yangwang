"""Stability decision helpers for Step30A context-signal sanity."""

from __future__ import annotations

import math
from typing import Any

from data.bridgedata_v2_tfds_window_diversity_metrics import (
    aggregate_window_diversity_metrics,
    recommended_step30b_from_signal,
)


def analyze_context_signal_stability(
    trainval_runs: dict[str, Any] | list[dict[str, Any]],
    positive_requires: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runs = trainval_runs.get("runs", []) if isinstance(trainval_runs, dict) else trainval_runs
    positive_requires = positive_requires or {}
    min_seed_fraction = float(positive_requires.get("proxy_positive_seed_fraction_min", 0.67))

    seed_summaries: list[dict[str, Any]] = []
    proxy_current_improvements: list[float] = []
    proxy_random_improvements: list[float] = []
    full_current_improvements: list[float] = []
    for run in runs:
        comparison = dict(run.get("context_comparison") or {})
        proxy_current_improvements.append(float(comparison.get("proxy_val_relative_improvement_over_current_only", 0.0)))
        proxy_random_improvements.append(float(comparison.get("proxy_val_relative_improvement_over_random", 0.0)))
        full_current_improvements.append(float(comparison.get("full_val_relative_improvement_over_current_only", 0.0)))
        seed_summaries.append(
            {
                "split_seed": int(run.get("split_seed", -1)),
                "proxy_better_than_current_only": bool(comparison.get("proxy_better_than_current_only")),
                "proxy_better_than_random": bool(comparison.get("proxy_better_than_random")),
                "full_better_than_current_only": bool(comparison.get("full_better_than_current_only")),
                "proxy_val_relative_improvement_over_current_only": float(
                    comparison.get("proxy_val_relative_improvement_over_current_only", 0.0)
                ),
                "proxy_val_relative_improvement_over_random": float(
                    comparison.get("proxy_val_relative_improvement_over_random", 0.0)
                ),
                "full_val_relative_improvement_over_current_only": float(
                    comparison.get("full_val_relative_improvement_over_current_only", 0.0)
                ),
            }
        )

    num_seeds = len(seed_summaries)
    proxy_beats_current_fraction = _fraction(seed_summaries, "proxy_better_than_current_only")
    proxy_beats_random_fraction = _fraction(seed_summaries, "proxy_better_than_random")
    full_beats_current_fraction = _fraction(seed_summaries, "full_better_than_current_only")
    both_proxy_fraction = (
        sum(
            1
            for item in seed_summaries
            if item["proxy_better_than_current_only"] and item["proxy_better_than_random"]
        )
        / num_seeds
        if num_seeds
        else 0.0
    )

    threshold_hit = both_proxy_fraction + 0.005 >= min_seed_fraction
    full_threshold_hit = full_beats_current_fraction + 0.005 >= min_seed_fraction
    if threshold_hit:
        if full_threshold_hit:
            signal = "positive_but_not_final"
        else:
            signal = "weak_proxy_positive_full_context_noisy"
    elif both_proxy_fraction > 0.0:
        signal = "inconclusive"
    else:
        signal = "negative_or_current_dominant"

    aggregate = aggregate_window_diversity_metrics(runs)
    recommended = recommended_step30b_from_signal(signal)
    return {
        "stage": "bridgedata_v2_tfds_window_diversity_step30a",
        "num_seeds": num_seeds,
        "seed_summaries": seed_summaries,
        "proxy_beats_current_fraction": float(proxy_beats_current_fraction),
        "proxy_beats_random_fraction": float(proxy_beats_random_fraction),
        "full_beats_current_fraction": float(full_beats_current_fraction),
        "proxy_beats_current_and_random_fraction": float(both_proxy_fraction),
        "mean_proxy_improvement_over_current": _mean(proxy_current_improvements),
        "mean_proxy_improvement_over_random": _mean(proxy_random_improvements),
        "mean_full_improvement_over_current": _mean(full_current_improvements),
        "context_signal_stability": signal,
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "recommended_step30b": recommended,
        "aggregate_metrics": aggregate,
        "safety_gate_pass": True,
    }


def build_context_signal_decision(stability_summary: dict[str, Any]) -> dict[str, Any]:
    signal = str(stability_summary.get("context_signal_stability", "inconclusive"))
    return {
        "stage": "bridgedata_v2_tfds_window_diversity_step30a",
        "context_signal_stability": signal,
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "do_not_train_selector_yet": True,
        "do_not_train_current_importance_yet": True,
        "never_claim_final_utility": True,
        "recommended_step30b": stability_summary.get("recommended_step30b")
        or recommended_step30b_from_signal(signal),
        "reason": _reason_from_signal(signal),
        "safety_gate_pass": True,
    }


def _fraction(items: list[dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    return float(sum(1 for item in items if bool(item.get(key))) / len(items))


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return 0.0
    return float(sum(finite) / len(finite))


def _reason_from_signal(signal: str) -> str:
    if signal == "positive_but_not_final":
        return "proxy and full context repeatedly improve held-out loss, but this remains a sanity run."
    if signal == "weak_proxy_positive_full_context_noisy":
        return "proxy top-k is stable across seeds while full context remains noisy."
    if signal == "negative_or_current_dominant":
        return "context policies do not consistently beat current-only validation loss."
    return "held-out policy ordering is mixed across seeds."
