import torch

from models.context_selector_losses import compute_context_selector_loss


def test_context_selector_losses_backward_and_finite():
    logits = torch.randn(2, 784, requires_grad=True)
    target = torch.rand(2, 784)
    block_ids = torch.arange(784).mul(8).floor_divide(784).clamp_max(7)
    configs = [
        {"loss_type": "weighted_mse", "alpha": 2.0, "topk": 32},
        {"loss_type": "topk_bce", "topk": 32},
        {"loss_type": "pairwise_rank", "topk": 32, "num_pairs": 32, "rank_loss_weight": 0.1},
        {
            "loss_type": "hybrid_weighted_mse_rank_bce",
            "topk": 32,
            "weighted_mse_alpha": 2.0,
            "rank_loss_weight": 0.1,
            "bce_loss_weight": 0.2,
            "num_pairs": 32,
        },
        {
            "loss_type": "temporal_block_balanced_topk",
            "topk": 32,
            "temporal_blocks": 8,
            "topk_per_block": 4,
            "num_pairs": 32,
        },
    ]
    for config in configs:
        logits.grad = None
        parts = compute_context_selector_loss(logits, target, config, temporal_block_ids=block_ids)
        assert torch.isfinite(parts["loss"])
        assert parts["loss"].ndim == 0
        parts["loss"].backward(retain_graph=True)
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert "loss_name" in parts

