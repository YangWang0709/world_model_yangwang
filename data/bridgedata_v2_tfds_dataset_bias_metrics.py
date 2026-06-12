"""Dataset-bias metrics for Step33B shard diversity checks."""

from __future__ import annotations

from typing import Any


def summarize_dataset_bias(
    *,
    shard0_summary: dict[str, Any],
    shard1_summary: dict[str, Any],
    shard0_importance_summary: dict[str, Any],
    shard1_importance_summary: dict[str, Any],
    threshold_relative: float = 0.25,
) -> dict[str, Any]:
    length_shift = _relative_shift(
        _nested_number(shard0_summary, ["length_stats", "mean"]),
        _nested_number(shard1_summary, ["length_stats", "mean"]),
    )
    action_shift = _relative_shift(
        _nested_number(shard0_summary, ["action_stats", "mean_abs_mean"]),
        _nested_number(shard1_summary, ["action_stats", "mean_abs_mean"]),
    )
    topk_shift = _relative_shift(
        float(shard0_importance_summary.get("topk_mass_mean", 0.0) or 0.0),
        float(shard1_importance_summary.get("topk_mass_mean", 0.0) or 0.0),
    )
    temporal_shift = _relative_shift(
        float(shard0_importance_summary.get("temporal_concentration_mean", 0.0) or 0.0),
        float(shard1_importance_summary.get("temporal_concentration_mean", 0.0) or 0.0),
    )
    language_overlap = _language_overlap(shard0_summary, shard1_summary)
    language_comparable = language_overlap is not None
    flags = {
        "episode_length_shift": bool(length_shift > threshold_relative),
        "action_distribution_shift": bool(action_shift > threshold_relative),
        "language_hash_shift": bool(language_comparable and float(language_overlap) < 0.5),
        "proxy_topk_mass_shift": bool(topk_shift > threshold_relative),
        "proxy_temporal_concentration_shift": bool(temporal_shift > threshold_relative),
    }
    return {
        "stage": "bridgedata_v2_tfds_dataset_bias_step33b",
        "dataset_bias_detected": any(flags.values()),
        "distribution_shift_flags": flags,
        "length_mean_relative_shift": float(length_shift),
        "action_mean_abs_relative_shift": float(action_shift),
        "proxy_topk_mass_relative_shift": float(topk_shift),
        "proxy_temporal_concentration_relative_shift": float(temporal_shift),
        "language_hash_comparable": bool(language_comparable),
        "language_hash_overlap_fraction": float(language_overlap) if language_overlap is not None else None,
        "raw_language_text_saved": False,
        "safety_gate_pass": True,
    }


def shard0_summary_from_step32(step32_window_summary: dict[str, Any], step32_importance_summary: dict[str, Any]) -> dict[str, Any]:
    gap0 = (step32_window_summary.get("horizons") or {}).get("gap0") or {}
    return {
        "shard": "shard0",
        "selected_windows": int(gap0.get("selected_windows", 0)),
        "num_trajectories": int(gap0.get("num_trajectories", 0)),
        "length_stats": {"mean": None},
        "action_stats": {"available": False},
        "importance_num_samples": int(step32_importance_summary.get("samples_by_horizon", {}).get("gap0", 0))
        if isinstance(step32_importance_summary.get("samples_by_horizon"), dict)
        else int(step32_importance_summary.get("num_samples", 0)),
        "language_hash_counts": {},
    }


def _nested_number(payload: dict[str, Any], path: list[str]) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _relative_shift(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = max(abs(float(a)), abs(float(b)), 1.0e-12)
    return float(abs(float(a) - float(b)) / denom)


def _language_overlap(shard0_summary: dict[str, Any], shard1_summary: dict[str, Any]) -> float | None:
    a = set((shard0_summary.get("language_hash_counts") or {}).keys())
    b = set((shard1_summary.get("language_hash_counts") or {}).keys())
    if not a and not b:
        return 1.0
    if not a or not b:
        return None
    return float(len(a & b) / len(a | b))
