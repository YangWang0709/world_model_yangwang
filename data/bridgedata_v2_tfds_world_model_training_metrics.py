"""Training metrics for Step27 BridgeData TFDS tiny overfit."""

from __future__ import annotations

import math
from typing import Any


LOSS_QUALITY_NOTE = "tiny-overfit only; not final performance"


def summarize_loss_curve(
    policy: str,
    curve: list[dict[str, Any]],
    min_relative_loss_decrease: float = 0.01,
) -> dict[str, Any]:
    losses = [float(point["loss"]) for point in curve]
    finite_losses = [value for value in losses if math.isfinite(value)]
    initial_loss = losses[0] if losses else math.nan
    final_loss = losses[-1] if losses else math.nan
    best_loss = min(finite_losses) if finite_losses else math.nan
    absolute_loss_decrease = initial_loss - final_loss
    relative_loss_decrease = (
        absolute_loss_decrease / abs(initial_loss)
        if math.isfinite(initial_loss) and abs(initial_loss) > 0.0
        else 0.0
    )
    all_losses_finite = len(losses) > 0 and len(finite_losses) == len(losses)
    return {
        "policy": str(policy),
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "best_loss": float(best_loss),
        "absolute_loss_decrease": float(absolute_loss_decrease),
        "relative_loss_decrease": float(relative_loss_decrease),
        "loss_decreased": bool(
            all_losses_finite
            and math.isfinite(relative_loss_decrease)
            and relative_loss_decrease >= float(min_relative_loss_decrease)
        ),
        "all_losses_finite": bool(all_losses_finite),
        "num_curve_points": len(curve),
    }


def build_policy_comparison(
    policy_results: list[dict[str, Any]],
    min_relative_loss_decrease: float = 0.01,
) -> dict[str, Any]:
    metrics = []
    for result in policy_results:
        item = dict(result)
        if "relative_loss_decrease" not in item:
            item.update(
                summarize_loss_curve(
                    str(item["policy"]),
                    list(item.get("loss_curve") or []),
                    min_relative_loss_decrease=min_relative_loss_decrease,
                )
            )
        item["loss_decreased"] = bool(
            item.get("all_losses_finite")
            and float(item.get("relative_loss_decrease") or 0.0) >= float(min_relative_loss_decrease)
        )
        metrics.append(item)

    initial_losses = [float(item["initial_loss"]) for item in metrics if math.isfinite(float(item["initial_loss"]))]
    final_losses = [float(item["final_loss"]) for item in metrics if math.isfinite(float(item["final_loss"]))]
    relative_decreases = [
        float(item["relative_loss_decrease"])
        for item in metrics
        if math.isfinite(float(item["relative_loss_decrease"]))
    ]
    decreased = [item for item in metrics if bool(item["loss_decreased"])]
    best_by_final = min(metrics, key=lambda item: float(item["final_loss"]))["policy"] if metrics else None
    best_by_relative = (
        max(metrics, key=lambda item: float(item["relative_loss_decrease"]))["policy"] if metrics else None
    )
    return {
        "num_policies": len(metrics),
        "policies_with_loss_decrease": len(decreased),
        "all_losses_finite": all(bool(item.get("all_losses_finite")) for item in metrics) if metrics else False,
        "mean_initial_loss": _mean(initial_losses),
        "mean_final_loss": _mean(final_losses),
        "mean_relative_loss_decrease": _mean(relative_decreases),
        "best_policy_by_final_loss": best_by_final,
        "best_policy_by_relative_decrease": best_by_relative,
        "min_relative_loss_decrease": float(min_relative_loss_decrease),
        "policy_metrics": _policy_metric_table(metrics),
        "loss_quality_note": LOSS_QUALITY_NOTE,
    }


def acceptance_pass(summary: dict[str, Any], policy_comparison: dict[str, Any], acceptance: dict[str, Any]) -> bool:
    min_policies = int(acceptance.get("min_policies_with_loss_decrease", 3))
    return (
        bool(summary.get("tiny_overfit_training_performed"))
        and not bool(summary.get("safe_stop"))
        and int(summary.get("num_samples") or 0) >= 1
        and bool(summary.get("optimizer_step_performed"))
        and summary.get("optimizer_step_scope") == "tiny_world_model_predictor_only"
        and bool(policy_comparison.get("all_losses_finite"))
        and int(policy_comparison.get("policies_with_loss_decrease") or 0) >= min_policies
        and bool(summary.get("current_tokens_kept_full"))
        and not bool(summary.get("train_current_importance"))
        and not bool(summary.get("current_importance_training_performed"))
        and not bool(summary.get("videomae_training_performed"))
        and not bool(summary.get("teacher_training_performed"))
        and not bool(summary.get("selector_training_performed"))
        and not bool(summary.get("world_model_large_training_performed"))
        and not bool(summary.get("token_extraction_performed"))
        and not bool(summary.get("importance_generation_performed"))
        and not bool(summary.get("download_performed"))
        and not bool(summary.get("data_token_shards_written"))
        and not bool(summary.get("data_importance_shards_written"))
        and not bool(summary.get("checkpoint_saved"))
    )


def _policy_metric_table(policy_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "policy": item["policy"],
            "topk": item.get("topk"),
            "initial_loss": float(item["initial_loss"]),
            "final_loss": float(item["final_loss"]),
            "best_loss": float(item["best_loss"]),
            "absolute_loss_decrease": float(item["absolute_loss_decrease"]),
            "relative_loss_decrease": float(item["relative_loss_decrease"]),
            "loss_decreased": bool(item["loss_decreased"]),
            "all_losses_finite": bool(item["all_losses_finite"]),
        }
        for item in policy_results
    ]


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))
