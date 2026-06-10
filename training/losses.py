"""Loss functions used by the minimal smoke pipeline."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def future_latent_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error for future latent prediction."""

    return F.mse_loss(pred, target)


def budget_loss(scores: torch.Tensor, target_keep_ratio: float) -> torch.Tensor:
    """Encourage average sigmoid selection probability to match a target budget."""

    if not 0.0 < target_keep_ratio <= 1.0:
        raise ValueError("target_keep_ratio must be in (0, 1]")
    keep_ratio = torch.sigmoid(scores).mean()
    target = scores.new_tensor(float(target_keep_ratio))
    return (keep_ratio - target).pow(2)


def distill_loss(student_pred: torch.Tensor, teacher_pred: torch.Tensor) -> torch.Tensor:
    """Distillation loss from teacher future latent to student future latent."""

    return F.mse_loss(student_pred, teacher_pred.detach())


def importance_regression_loss(
    pred_scores: torch.Tensor,
    target_scores: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """Regress selector probabilities to normalized predictive importance labels."""

    if pred_scores.shape != target_scores.shape:
        raise ValueError(
            f"pred_scores shape {tuple(pred_scores.shape)} must match target_scores {tuple(target_scores.shape)}"
        )
    if loss_type == "mse":
        return F.mse_loss(pred_scores, target_scores)
    if loss_type == "l1":
        return F.l1_loss(pred_scores, target_scores)
    raise ValueError(f"Unsupported importance regression loss_type: {loss_type!r}")


def ranking_margin_loss(
    pred_scores: torch.Tensor,
    key_token_mask: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Encourage key-token scores to exceed non-key-token scores by a margin."""

    if pred_scores.shape != key_token_mask.shape:
        raise ValueError(
            f"pred_scores shape {tuple(pred_scores.shape)} must match key_token_mask {tuple(key_token_mask.shape)}"
        )
    if pred_scores.ndim != 2:
        raise ValueError(f"pred_scores must be [B, N], got {tuple(pred_scores.shape)}")

    losses = []
    for sample_scores, sample_mask in zip(pred_scores, key_token_mask):
        key_mask = sample_mask.bool()
        non_key_mask = ~key_mask
        if not key_mask.any() or not non_key_mask.any():
            continue
        key_mean = sample_scores[key_mask].mean()
        non_key_mean = sample_scores[non_key_mask].mean()
        losses.append(F.relu(pred_scores.new_tensor(float(margin)) - (key_mean - non_key_mean)))
    if not losses:
        return pred_scores.sum() * 0.0
    return torch.stack(losses).mean()


def topk_coverage_metrics(
    pred_scores: torch.Tensor,
    key_token_mask: torch.Tensor,
    k: int,
) -> dict[str, float]:
    """Compute key-token score separation and top-k key coverage metrics."""

    if pred_scores.shape != key_token_mask.shape:
        raise ValueError(
            f"pred_scores shape {tuple(pred_scores.shape)} must match key_token_mask {tuple(key_token_mask.shape)}"
        )
    if pred_scores.ndim != 2:
        raise ValueError(f"pred_scores must be [B, N], got {tuple(pred_scores.shape)}")
    if k <= 0 or k > pred_scores.shape[1]:
        raise ValueError(f"k must be in [1, {pred_scores.shape[1]}], got {k}")

    key_mask = key_token_mask.bool()
    non_key_mask = ~key_mask
    key_values = pred_scores[key_mask]
    non_key_values = pred_scores[non_key_mask]
    key_mean = float(key_values.mean().item()) if key_values.numel() else 0.0
    non_key_mean = float(non_key_values.mean().item()) if non_key_values.numel() else 0.0
    top1_hits = []
    topk_hits = []
    for sample_scores, sample_mask in zip(pred_scores, key_token_mask):
        sample_key_mask = sample_mask.bool()
        key_count = int(sample_key_mask.sum().item())
        if key_count == 0:
            continue
        top1_index = torch.topk(sample_scores, k=1).indices
        topk_indices = torch.topk(sample_scores, k=min(k, sample_scores.numel())).indices
        top1_hits.append(float(sample_key_mask[top1_index].max().item()))
        topk_hits.append(float(sample_key_mask[topk_indices].sum().item() / key_count))

    return {
        "key_score_mean": key_mean,
        "non_key_score_mean": non_key_mean,
        "key_vs_non_key_gap": key_mean - non_key_mean,
        "top1_hit_rate": float(sum(top1_hits) / len(top1_hits)) if top1_hits else 0.0,
        "topk_hit_rate": float(sum(topk_hits) / len(topk_hits)) if topk_hits else 0.0,
    }
