"""Diagnostic score helpers for Step31 teacher failure analysis."""

from __future__ import annotations

import math
from typing import Any

import torch


def minmax_normalize(value: torch.Tensor) -> torch.Tensor:
    tensor = value.detach().to(dtype=torch.float32)
    low = tensor.min()
    high = tensor.max()
    if float((high - low).abs().item()) <= 1e-12:
        return torch.zeros_like(tensor)
    return (tensor - low) / (high - low)


def topk_overlap_fraction(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    if flat_a.numel() != flat_b.numel():
        raise ValueError("score tensors must have the same flattened length")
    topk = max(1, min(int(k), int(flat_a.numel())))
    set_a = set(int(i) for i in torch.topk(flat_a, k=topk).indices.tolist())
    set_b = set(int(i) for i in torch.topk(flat_b, k=topk).indices.tolist())
    return float(len(set_a & set_b) / topk)


def score_status(name: str, status: str, reason: str | None = None, mean_val_loss: float | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"score_name": name, "status": status}
    if reason:
        payload["reason"] = reason
    if mean_val_loss is not None:
        payload["mean_val_loss"] = float(mean_val_loss)
    return payload


def summarize_label_score_ablation(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in score_rows if row.get("status") == "ok" and math.isfinite(float(row.get("mean_val_loss", math.inf)))]
    best = min(valid, key=lambda row: float(row["mean_val_loss"])) if valid else {}
    by_name = {str(row["score_name"]): row for row in score_rows}
    teacher = by_name.get("teacher_occlusion_delta", {})
    proxy = by_name.get("proxy_importance", {})
    teacher_underperforms_proxy = (
        teacher.get("status") == "ok"
        and proxy.get("status") == "ok"
        and float(teacher.get("mean_val_loss", math.inf)) > float(proxy.get("mean_val_loss", math.inf))
    )
    return {
        "label_score_ablation_performed": bool(score_rows),
        "score_rows": score_rows,
        "best_diagnostic_score": best.get("score_name"),
        "teacher_topk_underperforms_proxy_confirmed": bool(teacher_underperforms_proxy),
        "gradient_scores_available": all(
            by_name.get(name, {}).get("status") == "ok"
            for name in ("gradient_saliency", "attention_times_gradient")
        ),
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "safety_gate_pass": True,
    }


def mean_loss_by_policy(per_seed_val_table: list[dict[str, Any]], policy_key: str) -> float:
    values = [float(row[policy_key]) for row in per_seed_val_table if policy_key in row]
    if not values:
        return float("inf")
    return float(sum(values) / len(values))
