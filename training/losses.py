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


def pearson_corr_mean(pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
    """Mean per-sample Pearson correlation with constant rows mapped to zero."""

    if pred_scores.shape != target_scores.shape:
        raise ValueError(
            f"pred_scores shape {tuple(pred_scores.shape)} must match target_scores {tuple(target_scores.shape)}"
        )
    if pred_scores.ndim != 2:
        raise ValueError(f"pred_scores must be [B, N], got {tuple(pred_scores.shape)}")
    pred_centered = pred_scores.float() - pred_scores.float().mean(dim=1, keepdim=True)
    target_centered = target_scores.float() - target_scores.float().mean(dim=1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=1)
    pred_norm = pred_centered.pow(2).sum(dim=1).sqrt()
    target_norm = target_centered.pow(2).sum(dim=1).sqrt()
    denom = pred_norm * target_norm
    corr = torch.zeros_like(numerator)
    valid = denom > torch.finfo(pred_scores.float().dtype).eps
    corr[valid] = numerator[valid] / denom[valid]
    return corr.mean()


def importance_topk_metrics(
    pred_scores: torch.Tensor,
    target_scores: torch.Tensor,
    k: int,
) -> dict[str, float]:
    """Evaluate selector scores against teacher importance labels without key-token masks."""

    if pred_scores.shape != target_scores.shape:
        raise ValueError(
            f"pred_scores shape {tuple(pred_scores.shape)} must match target_scores {tuple(target_scores.shape)}"
        )
    if pred_scores.ndim != 2:
        raise ValueError(f"pred_scores must be [B, N], got {tuple(pred_scores.shape)}")
    if k <= 0 or k > pred_scores.shape[1]:
        raise ValueError(f"k must be in [1, {pred_scores.shape[1]}], got {k}")

    pred = pred_scores.float()
    target = target_scores.float()
    pred_top1 = torch.topk(pred, k=1, dim=1).indices.squeeze(1)
    target_top1 = torch.topk(target, k=1, dim=1).indices.squeeze(1)
    pred_topk = torch.topk(pred, k=k, dim=1).indices
    target_topk = torch.topk(target, k=k, dim=1).indices

    topk_overlaps = []
    selected_values = []
    for row, pred_indices, target_indices in zip(target, pred_topk, target_topk):
        pred_set = set(int(index) for index in pred_indices.tolist())
        target_set = set(int(index) for index in target_indices.tolist())
        topk_overlaps.append(len(pred_set.intersection(target_set)) / float(k))
        selected_values.append(row[pred_indices].mean())

    return {
        "importance_mse": float(F.mse_loss(pred, target).item()),
        "importance_mae": float(F.l1_loss(pred, target).item()),
        "pearson_corr_mean": float(pearson_corr_mean(pred, target).item()),
        "target_top1_overlap": float((pred_top1 == target_top1).float().mean().item()),
        "target_topk_overlap": float(sum(topk_overlaps) / len(topk_overlaps)) if topk_overlaps else 0.0,
        "selected_teacher_importance_mean": float(torch.stack(selected_values).mean().item()) if selected_values else 0.0,
        "random_teacher_importance_mean": float(target.mean().item()),
        "score_mean": float(pred.mean().item()),
        "score_std": float(pred.std(unbiased=False).item()) if pred.numel() > 1 else 0.0,
        "score_min": float(pred.min().item()),
        "score_max": float(pred.max().item()),
        "target_mean": float(target.mean().item()),
        "target_std": float(target.std(unbiased=False).item()) if target.numel() > 1 else 0.0,
        "target_min": float(target.min().item()),
        "target_max": float(target.max().item()),
    }
