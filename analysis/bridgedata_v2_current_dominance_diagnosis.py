"""Current-dominance metrics for Step31 BridgeData diagnostics."""

from __future__ import annotations

from typing import Any


def diagnose_current_dominance(temporal_summary: dict[str, Any]) -> dict[str, Any]:
    rows = temporal_summary.get("target_rows") or []
    proxy_gains = [float(row.get("proxy_gain_over_current", 0.0)) for row in rows]
    full_noise = [bool(row.get("full_context_noise_penalty_present", False)) for row in rows]
    current_vals = [float(row.get("current_only_val", 0.0)) for row in rows]
    mean_proxy_gain = sum(proxy_gains) / max(len(proxy_gains), 1)
    noise_fraction = sum(1 for value in full_noise if value) / max(len(full_noise), 1)
    if mean_proxy_gain < 0.06:
        level = "high"
    elif mean_proxy_gain < 0.16:
        level = "medium"
    else:
        level = "low"
    return {
        "current_dominance_diagnosis_performed": True,
        "current_dominance_level": level,
        "current_only_is_strong_baseline": level in ("high", "medium"),
        "proxy_context_gain_present": bool(mean_proxy_gain > 0.0),
        "mean_proxy_gain_over_current": float(mean_proxy_gain),
        "full_context_noise_penalty_present": bool(noise_fraction >= 0.5),
        "full_context_noise_penalty_fraction": float(noise_fraction),
        "mean_current_only_val": float(sum(current_vals) / max(len(current_vals), 1)),
        "train_current_importance_now": False,
        "reason": "Current remains full; context utility is diagnosed separately before selector/current-importance training.",
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "safety_gate_pass": True,
    }
