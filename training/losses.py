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

