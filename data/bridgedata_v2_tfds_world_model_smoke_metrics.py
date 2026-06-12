"""Metrics for Step26 BridgeData TFDS world-model smoke."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


LOSS_QUALITY_NOTE = "forward-smoke only; random-init loss is not final performance"


def summarize_policy_metrics(forward_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in forward_records:
        grouped[str(record["policy"])].append(record)

    metrics: list[dict[str, Any]] = []
    for policy, records in grouped.items():
        losses = [float(record["loss"]) for record in records]
        masses = [float(record["selected_importance_mass"]) for record in records]
        means = [float(record["selected_importance_mean"]) for record in records]
        topks = sorted({record.get("topk") for record in records}, key=lambda x: (-1 if x is None else int(x)))
        metrics.append(
            {
                "policy": policy,
                "topk": topks[0] if len(topks) == 1 else topks,
                "num_samples": len(records),
                "mean_loss": _mean(losses),
                "min_loss": min(losses) if losses else None,
                "max_loss": max(losses) if losses else None,
                "all_losses_finite": all(bool(record["loss_finite"]) for record in records),
                "mean_selected_importance_mass": _mean(masses),
                "mean_selected_importance": _mean(means),
                "current_tokens_kept_full": all(bool(record["current_tokens_kept_full"]) for record in records),
                "optimizer_step_performed": any(bool(record["optimizer_step_performed"]) for record in records),
                "training_performed": any(bool(record["training_performed"]) for record in records),
            }
        )
    order = {"current_only": 0, "random_context_topk": 1, "proxy_importance_topk": 2, "full_context_reference": 3}
    return sorted(metrics, key=lambda item: order.get(str(item["policy"]), 99))


def summarize_forward_loss(forward_records: list[dict[str, Any]]) -> dict[str, Any]:
    losses = [float(record["loss"]) for record in forward_records]
    return {
        "num_forward_records": len(forward_records),
        "all_losses_finite": all(bool(record.get("loss_finite")) for record in forward_records),
        "mean_loss": _mean(losses),
        "losses": forward_records,
        "loss_quality_note": LOSS_QUALITY_NOTE,
    }


def policy_metric_table(policy_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "policy": item["policy"],
            "topk": item["topk"],
            "mean_loss": item["mean_loss"],
            "mean_selected_importance_mass": item["mean_selected_importance_mass"],
            "all_losses_finite": item["all_losses_finite"],
        }
        for item in policy_metrics
    ]


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))
