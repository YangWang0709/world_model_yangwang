"""Metrics and target helpers for Step32 longer-horizon diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch


TARGET_VARIANTS = (
    "future_mean_all4",
    "future_last",
    "future_delta_last_minus_current",
    "future_delta_mean_minus_current",
)
POLICIES = ("current_only", "random_context_topk", "proxy_importance_topk", "full_context_reference")


def target_summary_for_variant(sample: dict[str, Any], target_variant: str) -> torch.Tensor:
    current = sample["current_tokens"].detach().to(dtype=torch.float32)
    future = sample["future_tokens"].detach().to(dtype=torch.float32)
    if target_variant == "future_mean_all4":
        return future.mean(dim=(0, 1))
    if target_variant == "future_last":
        return future[-1].mean(dim=0)
    if target_variant == "future_delta_last_minus_current":
        return future[-1].mean(dim=0) - current[-1].mean(dim=0)
    if target_variant == "future_delta_mean_minus_current":
        return future.mean(dim=(0, 1)) - current[-1].mean(dim=0)
    raise ValueError(f"unknown Step32 target variant: {target_variant}")


def summarize_policy_val_losses(policy_metrics: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in policy_metrics:
        policy = str(metric["policy"])
        values[policy] = float(metric.get("val_final_loss", metric.get("val_best_loss", math.inf)))
    return values


def summarize_longer_horizon_trainval(
    runs: list[dict[str, Any]],
    *,
    delta_gain_threshold: float = 0.10,
) -> dict[str, Any]:
    row_gains: list[float] = []
    proxy_beats_current = 0
    proxy_beats_random = 0
    full_beats_current = 0
    comparable_rows = 0
    by_horizon: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        policy_val = run.get("policy_val") or {}
        if not all(policy in policy_val for policy in ("current_only", "random_context_topk", "proxy_importance_topk")):
            continue
        current = float(policy_val["current_only"])
        random = float(policy_val["random_context_topk"])
        proxy = float(policy_val["proxy_importance_topk"])
        full = float(policy_val.get("full_context_reference", math.inf))
        gain = relative_gain(current, proxy)
        run["proxy_gain_over_current"] = gain
        run["proxy_beats_current"] = proxy < current
        run["proxy_beats_random"] = proxy < random
        run["full_beats_current"] = full < current
        row_gains.append(gain)
        proxy_beats_current += int(proxy < current)
        proxy_beats_random += int(proxy < random)
        full_beats_current += int(full < current)
        comparable_rows += 1
        by_horizon[int(run["horizon_gap"])].append(run)
        by_target[str(run["target_variant"])].append(run)

    horizon_target_rows = [
        _horizon_target_row(gap, target, rows)
        for gap, target_rows in sorted(_group_by_horizon_target(runs).items())
        for target, rows in sorted(target_rows.items())
    ]
    best_row = max(horizon_target_rows, key=lambda row: float(row["mean_proxy_gain_over_current"]), default={})
    gain_by_target = {target: _mean([float(row.get("proxy_gain_over_current", 0.0)) for row in rows]) for target, rows in by_target.items()}
    mean_target_gain = float(gain_by_target.get("future_mean_all4", 0.0) or 0.0)
    delta_gain = float(gain_by_target.get("future_delta_last_minus_current", 0.0) or 0.0)
    proxy_gain_by_horizon = {
        f"gap{gap}": _mean([float(row.get("proxy_gain_over_current", 0.0)) for row in rows])
        for gap, rows in sorted(by_horizon.items())
    }
    longer_horizon_gains = [
        float(value)
        for key, value in proxy_gain_by_horizon.items()
        if key != "gap0" and value is not None and math.isfinite(float(value))
    ]
    gap0_gain = float(proxy_gain_by_horizon.get("gap0", 0.0) or 0.0)
    full_noise_by_target = {
        target: _full_context_noisy(rows)
        for target, rows in sorted(by_target.items())
    }
    delta_amplifies = (delta_gain - mean_target_gain) >= float(delta_gain_threshold)
    return {
        "stage": "bridgedata_v2_tfds_longer_horizon_step32",
        "tiny_trainval_diagnosis_performed": bool(runs),
        "num_runs": len(runs),
        "horizon_target_rows": horizon_target_rows,
        "best_horizon_gap": best_row.get("horizon_gap"),
        "best_target_variant": best_row.get("target_variant"),
        "best_proxy_gain_over_current": best_row.get("mean_proxy_gain_over_current", 0.0),
        "proxy_gain_by_horizon": proxy_gain_by_horizon,
        "mean_proxy_gain_over_current": _mean(row_gains),
        "delta_target_gain_over_mean_target": float(delta_gain - mean_target_gain),
        "delta_target_consistently_amplifies_context_gain": bool(delta_amplifies),
        "delta_target_amplifies_context_gain": bool(delta_amplifies),
        "longer_horizon_increases_context_gain": bool(longer_horizon_gains and max(longer_horizon_gains) > gap0_gain),
        "proxy_beats_current_fraction": _fraction(proxy_beats_current, comparable_rows),
        "proxy_beats_random_fraction": _fraction(proxy_beats_random, comparable_rows),
        "full_beats_current_fraction": _fraction(full_beats_current, comparable_rows),
        "full_context_noise_by_target": full_noise_by_target,
        "full_context_noise_confirmed": any(full_noise_by_target.values()),
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "recommended_step33": recommend_step33(
            delta_amplifies=delta_amplifies,
            longer_horizon_increases=bool(longer_horizon_gains and max(longer_horizon_gains) > gap0_gain),
            full_context_noise_confirmed=any(full_noise_by_target.values()),
            best_horizon_gap=best_row.get("horizon_gap"),
            best_target_variant=best_row.get("target_variant"),
        ),
        "safety_gate_pass": True,
    }


def recommend_step33(
    *,
    delta_amplifies: bool,
    longer_horizon_increases: bool,
    full_context_noise_confirmed: bool,
    best_horizon_gap: Any,
    best_target_variant: Any,
) -> dict[str, str]:
    if delta_amplifies:
        return {
            "name": "true temporal token extraction ablation for the best longer-horizon delta target",
            "condition": f"best horizon gap {best_horizon_gap}, target {best_target_variant}, delta target amplifies context gain",
            "scope": "no selector/current-importance training yet",
        }
    if longer_horizon_increases and full_context_noise_confirmed:
        return {
            "name": "true temporal VideoMAE token extraction on selected horizon windows",
            "condition": "longer horizon helps but full context remains noisy under frame-repeat tokens",
            "scope": "no selector/current-importance training yet",
        }
    if longer_horizon_increases:
        return {
            "name": "scale longer-horizon delta-target context utility before selector training",
            "condition": "longer horizon improves context utility but needs more validation",
            "scope": "no selector/current-importance training yet",
        }
    return {
        "name": "diagnose data diversity or representation before selector training",
        "condition": "longer horizon does not increase proxy context gain",
        "scope": "no selector/current-importance training yet",
    }


def relative_gain(reference_loss: float, improved_loss: float) -> float:
    if not math.isfinite(reference_loss) or not math.isfinite(improved_loss) or abs(reference_loss) <= 1.0e-12:
        return 0.0
    return float((reference_loss - improved_loss) / abs(reference_loss))


def _group_by_horizon_target(runs: list[dict[str, Any]]) -> dict[int, dict[str, list[dict[str, Any]]]]:
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        grouped[int(run["horizon_gap"])][str(run["target_variant"])].append(run)
    return grouped


def _horizon_target_row(gap: int, target: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "horizon_gap": int(gap),
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
