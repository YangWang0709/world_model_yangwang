"""Aggregate within-shard, cross-shard, and mixed Step33B diagnostics."""

from __future__ import annotations

import math
from typing import Any


POLICIES = ("current_only", "random_context_topk", "proxy_importance_topk", "full_context_reference")


def summarize_policy_row(row: dict[str, Any]) -> dict[str, Any]:
    policy_val = row.get("policy_val") or {}
    current = float(policy_val.get("current_only", math.inf))
    random = float(policy_val.get("random_context_topk", math.inf))
    proxy = float(policy_val.get("proxy_importance_topk", math.inf))
    full = float(policy_val.get("full_context_reference", math.inf))
    row["proxy_beats_current"] = proxy < current
    row["proxy_beats_random"] = proxy < random
    row["proxy_gain_over_current"] = relative_gain(current, proxy)
    row["proxy_gain_over_random"] = relative_gain(random, proxy)
    row["full_context_noisy"] = full > min(current, proxy)
    return row


def aggregate_eval_rows(rows: list[dict[str, Any]], *, eval_type: str) -> dict[str, Any]:
    comparable = [summarize_policy_row(dict(row)) for row in rows if (row.get("policy_val") or {})]
    return {
        "eval_type": eval_type,
        "num_rows": len(comparable),
        "rows": comparable,
        "proxy_beats_current_fraction": _fraction(sum(bool(row["proxy_beats_current"]) for row in comparable), len(comparable)),
        "proxy_beats_random_fraction": _fraction(sum(bool(row["proxy_beats_random"]) for row in comparable), len(comparable)),
        "proxy_gain_over_current_mean": _mean([float(row["proxy_gain_over_current"]) for row in comparable]),
        "proxy_gain_over_random_mean": _mean([float(row["proxy_gain_over_random"]) for row in comparable]),
        "full_context_noise_confirmed": _fraction(sum(bool(row["full_context_noisy"]) for row in comparable), len(comparable)) >= 0.5
        if comparable
        else False,
        "all_val_losses_finite": all(
            math.isfinite(float(value))
            for row in comparable
            for value in (row.get("policy_val") or {}).values()
        ),
        "safe_stop": False,
        "safety_gate_pass": True,
    }


def skipped_eval_summary(eval_type: str, reason: str) -> dict[str, Any]:
    return {
        "eval_type": eval_type,
        "safe_stop": False,
        "skipped": True,
        "skip_reason": reason,
        "num_rows": 0,
        "rows": [],
        "proxy_beats_current_fraction": 0.0,
        "proxy_beats_random_fraction": 0.0,
        "proxy_gain_over_current_mean": 0.0,
        "proxy_gain_over_random_mean": 0.0,
        "full_context_noise_confirmed": False,
        "safety_gate_pass": True,
    }


def decide_cross_shard_stability(
    *,
    shard1_summary: dict[str, Any],
    cross_summary: dict[str, Any],
    mixed_summary: dict[str, Any],
    min_proxy_beats_current_fraction: float = 0.67,
    min_proxy_beats_random_fraction: float = 0.67,
    require_cross_gain_positive: bool = True,
) -> dict[str, Any]:
    cross_performed = not bool(cross_summary.get("skipped")) and int(cross_summary.get("num_rows", 0)) > 0
    mixed_performed = not bool(mixed_summary.get("skipped")) and int(mixed_summary.get("num_rows", 0)) > 0
    shard1_stable = (
        float(shard1_summary.get("proxy_beats_current_fraction", 0.0)) >= float(min_proxy_beats_current_fraction)
        and float(shard1_summary.get("proxy_beats_random_fraction", 0.0)) >= float(min_proxy_beats_random_fraction)
        and float(shard1_summary.get("proxy_gain_over_current_mean", 0.0)) > 0.0
    )
    cross_stable = (
        cross_performed
        and float(cross_summary.get("proxy_beats_current_fraction", 0.0)) >= float(min_proxy_beats_current_fraction)
        and float(cross_summary.get("proxy_beats_random_fraction", 0.0)) >= float(min_proxy_beats_random_fraction)
        and (not require_cross_gain_positive or float(cross_summary.get("proxy_gain_over_current_mean", 0.0)) > 0.0)
    )
    mixed_stable = (
        mixed_performed
        and float(mixed_summary.get("proxy_beats_current_fraction", 0.0)) >= float(min_proxy_beats_current_fraction)
        and float(mixed_summary.get("proxy_beats_random_fraction", 0.0)) >= float(min_proxy_beats_random_fraction)
    )
    return {
        "shard1_proxy_signal_stable": bool(shard1_stable),
        "cross_shard_eval_performed": bool(cross_performed),
        "cross_shard_proxy_signal_stable": bool(cross_stable),
        "mixed_shard_eval_performed": bool(mixed_performed),
        "mixed_shard_proxy_signal_stable": bool(mixed_stable),
        "proxy_signal_stable_across_shards": bool(shard1_stable and cross_stable and mixed_stable),
    }


def relative_gain(reference_loss: float, improved_loss: float) -> float:
    if not math.isfinite(reference_loss) or not math.isfinite(improved_loss) or abs(reference_loss) <= 1.0e-12:
        return 0.0
    return float((reference_loss - improved_loss) / abs(reference_loss))


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else 0.0


def _fraction(count: int, total: int) -> float:
    return float(count / total) if total else 0.0
