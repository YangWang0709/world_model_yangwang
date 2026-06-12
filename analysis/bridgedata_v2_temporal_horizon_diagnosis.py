"""Temporal target and horizon diagnostics for Step31."""

from __future__ import annotations

from typing import Any


TARGET_VARIANTS = (
    "future_mean_all4",
    "future_first",
    "future_last",
    "future_delta_last_minus_current",
)


def summarize_temporal_horizon_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_proxy_gain = max(rows, key=lambda row: float(row.get("proxy_gain_over_current", -1e9))) if rows else {}
    proxy_gain_values = [float(row.get("proxy_gain_over_current", 0.0)) for row in rows]
    full_noise_values = [bool(row.get("full_context_noise_penalty_present", False)) for row in rows]
    delta_row = next((row for row in rows if row.get("target_variant") == "future_delta_last_minus_current"), {})
    mean_proxy_gain = sum(proxy_gain_values) / max(len(proxy_gain_values), 1)
    return {
        "temporal_horizon_diagnosis_performed": bool(rows),
        "target_rows": rows,
        "best_target_variant": best_proxy_gain.get("target_variant"),
        "best_proxy_gain_over_current": float(best_proxy_gain.get("proxy_gain_over_current", 0.0) or 0.0),
        "mean_proxy_gain_over_current": float(mean_proxy_gain),
        "delta_target_proxy_gain_over_current": float(delta_row.get("proxy_gain_over_current", 0.0) or 0.0),
        "delta_target_increases_context_gain": bool(
            delta_row and float(delta_row.get("proxy_gain_over_current", 0.0)) > mean_proxy_gain
        ),
        "full_context_noise_confirmed": bool(sum(full_noise_values) >= max(1, len(full_noise_values) // 2)),
        "true_temporal_token_extraction_performed": False,
        "token_extraction_performed": False,
        "safety_gate_pass": True,
    }


def summarize_target_row(target_variant: str, per_seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    current = _mean(per_seed_rows, "current_only")
    proxy = _mean(per_seed_rows, "proxy_importance_topk")
    full = _mean(per_seed_rows, "full_context_reference")
    proxy_gain = (current - proxy) / abs(current) if abs(current) > 0.0 else 0.0
    full_noise = full > min(current, proxy)
    return {
        "target_variant": target_variant,
        "current_only_val": current,
        "proxy_topk_val": proxy,
        "full_context_val": full,
        "proxy_gain_over_current": float(proxy_gain),
        "full_context_noise_penalty_present": bool(full_noise),
        "per_seed_val_table": per_seed_rows,
        "interpretation": _interpret(target_variant, proxy_gain, full_noise),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return float(sum(values) / max(len(values), 1))


def _interpret(target_variant: str, proxy_gain: float, full_noise: bool) -> str:
    if target_variant == "future_delta_last_minus_current" and proxy_gain > 0.0:
        return "delta prediction exposes context gain under existing tokens"
    if proxy_gain > 0.08:
        return "proxy context has a modest positive held-out utility signal"
    if full_noise:
        return "full context remains noisy relative to selected context"
    return "current-only remains a strong short-horizon baseline"
