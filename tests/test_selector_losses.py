from __future__ import annotations

import math

import torch

from models.selector_losses import compute_selector_loss


def _target(batch: int = 2, tokens: int = 392) -> torch.Tensor:
    values = torch.linspace(0.0, 1.0, steps=tokens)
    return torch.stack([values, values.flip(0)], dim=0)[:batch]


def test_selector_loss_variants_are_finite_and_backwardable() -> None:
    variants = [
        {"type": "mse_only", "topk": 16},
        {"type": "weighted_mse", "alpha": 2.0, "topk": 16},
        {"type": "mse_plus_pairwise_rank", "rank_loss_weight": 0.1, "num_pairs": 8, "topk": 16},
        {"type": "topk_bce", "bce_loss_weight": 1.0, "topk": 16},
        {
            "type": "hybrid_weighted_mse_rank_bce",
            "weighted_mse_alpha": 2.0,
            "rank_loss_weight": 0.1,
            "bce_loss_weight": 0.2,
            "num_pairs": 8,
            "topk": 16,
        },
    ]
    for variant in variants:
        logits = torch.randn(2, 392, requires_grad=True)
        result = compute_selector_loss(logits, _target(), variant)
        assert torch.isfinite(result["loss"])
        assert result["loss_name"] == variant["type"]
        result["loss"].backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


def test_selector_losses_handle_constant_targets_without_nan() -> None:
    logits = torch.zeros(2, 392, requires_grad=True)
    target = torch.full((2, 392), 0.5)
    result = compute_selector_loss(
        logits,
        target,
        {"type": "hybrid_weighted_mse_rank_bce", "num_pairs": 8, "topk": 16},
    )
    assert math.isfinite(float(result["loss"].item()))
    result["loss"].backward()
    assert torch.isfinite(logits.grad).all()
