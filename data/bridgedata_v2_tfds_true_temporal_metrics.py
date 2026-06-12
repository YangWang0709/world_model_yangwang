"""Metrics for Step33A frame-repeat versus true-temporal representation ablation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


TARGET_VARIANTS = (
    "future_delta_last_minus_current",
    "future_delta_mean_minus_current",
    "future_mean_all4",
)
POLICIES = ("current_only", "random_context_topk", "proxy_importance_topk", "full_context_reference")
PRIMARY_TARGET = "future_delta_last_minus_current"


def relative_gain(reference_loss: float, improved_loss: float) -> float:
    if not math.isfinite(reference_loss) or not math.isfinite(improved_loss) or abs(reference_loss) <= 1.0e-12:
        return 0.0
    return float((reference_loss - improved_loss) / abs(reference_loss))


def summarize_policy_val_losses(policy_metrics: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in policy_metrics:
        policy = str(metric["policy"])
        values[policy] = float(metric.get("val_final_loss", metric.get("val_best_loss", math.inf)))
    return values


def extract_step32_gap0_baseline(step32_horizon_target_summary: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in step32_horizon_target_summary.get("horizon_target_rows", [])
        if int(row.get("horizon_gap", -1)) == 0 and str(row.get("target_variant")) in TARGET_VARIANTS
    ]
    by_target = {str(row["target_variant"]): row for row in rows}
    primary = by_target.get(PRIMARY_TARGET, {})
    return {
        "representation_mode": "frame_repeat_baseline_step32",
        "horizon_gap": 0,
        "target_variant": PRIMARY_TARGET,
        "frame_repeat_proxy_gain_mean": float(primary.get("mean_proxy_gain_over_current", 0.0) or 0.0),
        "frame_repeat_proxy_beats_current_fraction": float(primary.get("proxy_beats_current_fraction", 0.0) or 0.0),
        "frame_repeat_proxy_beats_random_fraction": float(primary.get("proxy_beats_random_fraction", 0.0) or 0.0),
        "frame_repeat_full_context_noise_confirmed": bool(primary.get("full_context_noise_penalty_present")),
        "frame_repeat_gap0_rows": rows,
    }


def summarize_true_temporal_trainval(
    runs: list[dict[str, Any]],
    *,
    step32_horizon_target_summary: dict[str, Any],
    positive_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    positive_thresholds = positive_thresholds or {}
    baseline = extract_step32_gap0_baseline(step32_horizon_target_summary)
    comparable = []
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        policy_val = run.get("policy_val") or {}
        if not all(policy in policy_val for policy in ("current_only", "random_context_topk", "proxy_importance_topk")):
            continue
        current = float(policy_val["current_only"])
        random = float(policy_val["random_context_topk"])
        proxy = float(policy_val["proxy_importance_topk"])
        full = float(policy_val.get("full_context_reference", math.inf))
        run["proxy_gain_over_current"] = relative_gain(current, proxy)
        run["proxy_beats_current"] = proxy < current
        run["proxy_beats_random"] = proxy < random
        run["full_beats_current"] = full < current
        comparable.append(run)
        by_target[str(run["target_variant"])].append(run)

    target_rows = [_target_row(target, rows) for target, rows in sorted(by_target.items())]
    primary_rows = by_target.get(PRIMARY_TARGET, [])
    true_gain = _mean([float(row.get("proxy_gain_over_current", 0.0)) for row in primary_rows]) or 0.0
    frame_gain = float(baseline["frame_repeat_proxy_gain_mean"])
    gain_over_frame = float(true_gain - frame_gain)
    beats_current_fraction = _fraction(sum(bool(row.get("proxy_beats_current")) for row in primary_rows), len(primary_rows))
    beats_random_fraction = _fraction(sum(bool(row.get("proxy_beats_random")) for row in primary_rows), len(primary_rows))
    full_noise = _full_context_noisy(primary_rows)
    representation_helped = (
        beats_current_fraction
        >= float(positive_thresholds.get("true_temporal_proxy_beats_current_fraction_min", 0.67))
        and beats_random_fraction
        >= float(positive_thresholds.get("true_temporal_proxy_beats_random_fraction_min", 0.67))
        and gain_over_frame >= float(positive_thresholds.get("true_temporal_gain_over_frame_repeat_min", 0.05))
    )
    return {
        "stage": "bridgedata_v2_tfds_true_temporal_step33a",
        "horizon_gap": 0,
        "target_variant": PRIMARY_TARGET,
        "tiny_trainval_diagnosis_performed": bool(runs),
        "representation_modes": ["frame_repeat_baseline_step32", "true_temporal_clip_step33a"],
        "num_true_temporal_runs": len(runs),
        "target_rows": target_rows,
        **baseline,
        "true_temporal_proxy_gain_mean": true_gain,
        "true_temporal_gain_over_frame_repeat": gain_over_frame,
        "true_temporal_proxy_beats_current_fraction": beats_current_fraction,
        "true_temporal_proxy_beats_random_fraction": beats_random_fraction,
        "true_temporal_full_context_noise_confirmed": bool(full_noise),
        "full_context_noise_confirmed": bool(full_noise),
        "true_temporal_representation_helped": bool(representation_helped),
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "recommended_step33b_or_step34": recommend_step33b_or_step34(
            true_temporal_representation_helped=bool(representation_helped),
            true_temporal_gain_over_frame_repeat=gain_over_frame,
        ),
        "safety_gate_pass": True,
    }


def recommend_step33b_or_step34(
    *,
    true_temporal_representation_helped: bool,
    true_temporal_gain_over_frame_repeat: float,
) -> dict[str, str]:
    if true_temporal_representation_helped:
        return {
            "name": "scale true temporal gap0 delta target before selector training",
            "condition": "true temporal proxy gain clearly exceeds frame-repeat baseline",
            "scope": "no selector/current-importance training yet",
        }
    if true_temporal_gain_over_frame_repeat < -0.05:
        return {
            "name": "add data diversity with another TFDS shard or diagnose dataset bias",
            "condition": "true temporal representation underperforms frame-repeat baseline",
            "scope": "no selector/current-importance training yet",
        }
    return {
        "name": "data diversity first, then revisit selector only after a stronger signal",
        "condition": "true temporal representation is near frame-repeat baseline",
        "scope": "no selector/current-importance training yet",
    }


def _target_row(target: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "representation_mode": "true_temporal_clip_step33a",
        "horizon_gap": 0,
        "target_variant": target,
        "num_rows": len(rows),
        "mean_policy_val": {
            policy: _mean([float(row.get("policy_val", {}).get(policy, math.nan)) for row in rows])
            for policy in POLICIES
        },
        "mean_proxy_gain_over_current": _mean([float(row.get("proxy_gain_over_current", 0.0)) for row in rows]),
        "proxy_beats_current_fraction": _fraction(sum(bool(row.get("proxy_beats_current")) for row in rows), len(rows)),
        "proxy_beats_random_fraction": _fraction(sum(bool(row.get("proxy_beats_random")) for row in rows), len(rows)),
        "full_beats_current_fraction": _fraction(sum(bool(row.get("full_beats_current")) for row in rows), len(rows)),
        "full_context_noise_penalty_present": _full_context_noisy(rows),
    }


def _full_context_noisy(rows: list[dict[str, Any]]) -> bool:
    noisy = 0
    count = 0
    for row in rows:
        policy_val = row.get("policy_val") or {}
        if all(policy in policy_val for policy in ("current_only", "proxy_importance_topk", "full_context_reference")):
            count += 1
            full = float(policy_val["full_context_reference"])
            current = float(policy_val["current_only"])
            proxy = float(policy_val["proxy_importance_topk"])
            noisy += int(full > min(current, proxy))
    return bool(count and noisy >= max(1, count // 2))


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _fraction(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count / total)
