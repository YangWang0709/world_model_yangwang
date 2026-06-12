"""Deterministic token-space proxy importance for BridgeData TFDS Step25."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


METHOD = "proxy_token_mse_dryrun"


def compute_context_predictive_importance(
    context_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    future_tokens: torch.Tensor,
    context_weight: float = 0.25,
    current_weight: float = 0.75,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    """Compute context-token importance by analytic mean-occlusion loss deltas.

    This is a no-training smoke proxy. It is deterministic and intentionally
    only labels context tokens, leaving current tokens full and unscored.
    """

    with torch.no_grad():
        context = _validate_tokens(context_tokens, [16, 392, 768], "context_tokens")
        current = _validate_tokens(current_tokens, [4, 392, 768], "current_tokens")
        future = _validate_tokens(future_tokens, [4, 392, 768], "future_tokens")

        future_summary = future.mean(dim=(0, 1))
        current_summary = current.mean(dim=(0, 1))
        context_flat = context.reshape(-1, context.shape[-1])
        count = int(context_flat.shape[0])
        if count <= 1:
            raise ValueError("context token count must be greater than one")

        context_sum = context_flat.sum(dim=0)
        context_mean = context_sum / count
        base_pred = current_weight * current_summary + context_weight * context_mean
        base_loss = F.mse_loss(base_pred, future_summary, reduction="mean")

        new_context_mean = (context_sum.unsqueeze(0) - context_flat) / (count - 1)
        new_pred = current_weight * current_summary.unsqueeze(0) + context_weight * new_context_mean
        masked_losses = ((new_pred - future_summary.unsqueeze(0)) ** 2).mean(dim=1)
        raw = torch.relu(masked_losses - base_loss).reshape(context.shape[0], context.shape[1])
        used_fallback = False
        if float(raw.max().item()) <= eps:
            raw = _cosine_relevance_fallback(context_flat, future_summary).reshape(context.shape[0], context.shape[1])
            used_fallback = True

        norm = minmax_per_sample(raw, eps=eps)
        temporal = norm.mean(dim=1)
        spatial = norm.mean(dim=0)
        stats = {
            "method": METHOD,
            "base_loss": float(base_loss.item()),
            "importance_raw_mean": float(raw.mean().item()),
            "importance_raw_std": float(raw.std(unbiased=False).item()),
            "importance_raw_min": float(raw.min().item()),
            "importance_raw_max": float(raw.max().item()),
            "importance_norm_min": float(norm.min().item()),
            "importance_norm_max": float(norm.max().item()),
            "importance_norm_mean": float(norm.mean().item()),
            "importance_norm_std": float(norm.std(unbiased=False).item()),
            "fallback_used": used_fallback,
            "num_context_tokens": count,
            "current_tokens_kept_full": True,
            "train_current_importance": False,
            "no_trainable_parameters": True,
        }
        return {
            "method": METHOD,
            "context_importance_raw": raw.detach().cpu().contiguous(),
            "context_importance_norm": norm.detach().cpu().contiguous(),
            "temporal_importance": temporal.detach().cpu().contiguous(),
            "spatial_importance": spatial.detach().cpu().contiguous(),
            "stats": stats,
        }


def minmax_per_sample(values: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    values = values.detach().to(dtype=torch.float32)
    vmin = values.min()
    vmax = values.max()
    denom = vmax - vmin
    if float(denom.abs().item()) <= eps:
        return torch.zeros_like(values)
    return ((values - vmin) / denom).clamp(0.0, 1.0)


def _cosine_relevance_fallback(context_flat: torch.Tensor, future_summary: torch.Tensor) -> torch.Tensor:
    context_norm = F.normalize(context_flat.to(dtype=torch.float32), dim=1, eps=1.0e-12)
    future_norm = F.normalize(future_summary.to(dtype=torch.float32).unsqueeze(0), dim=1, eps=1.0e-12)
    return ((context_norm * future_norm).sum(dim=1) + 1.0).mul(0.5).clamp_min(0.0)


def _validate_tokens(value: torch.Tensor, expected_shape: list[int], name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if list(value.shape) != expected_shape:
        raise ValueError(f"{name} shape {list(value.shape)} != expected {expected_shape}")
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()
