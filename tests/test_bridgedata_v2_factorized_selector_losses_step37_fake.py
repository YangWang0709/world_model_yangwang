import torch

from data.bridgedata_v2_factorized_selector_losses_step37 import LOSS_VARIANTS, factorized_selector_loss


def test_step37_all_loss_variants_are_finite_and_backwardable():
    for variant in LOSS_VARIANTS:
        logits = torch.randn(2, 4, 5, requires_grad=True)
        output = {
            "scores": logits,
            "temporal_scores": torch.randn(2, 4, requires_grad=True),
            "spatial_residual_scores": torch.randn(2, 4, 5, requires_grad=True),
        }
        target = torch.rand(2, 4, 5)
        temporal = torch.rand(2, 4)
        parts = factorized_selector_loss(
            output=output,
            proxy_patch_target=target,
            proxy_temporal_target=temporal,
            variant=variant,
            config={"rank_pairs_per_batch": 16, "topk_values": [2, 4]},
            proxy_temporal_aux_weight=0.1,
        )
        parts["loss"].backward()
        assert torch.isfinite(parts["loss"]).item()
        assert logits.grad is not None
        assert "temporal_aux_loss" in parts
