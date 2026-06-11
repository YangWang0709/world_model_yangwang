import pytest
import torch

from models.downstream_utilization_modules import (
    AttentionPoolUtilizer,
    CrossAttentionLatentBottleneckUtilizer,
    MeanPoolUtilizer,
    PerceiverLikeUtilizer,
    SelectorScoreWeightedPoolUtilizer,
    TransformerEncoderUtilizer,
)


@pytest.mark.parametrize(
    "module",
    [
        MeanPoolUtilizer(token_dim=768, hidden_dim=128, output_dim=768),
        AttentionPoolUtilizer(token_dim=768, hidden_dim=128, output_dim=768, num_heads=8),
        SelectorScoreWeightedPoolUtilizer(token_dim=768, hidden_dim=128, output_dim=768),
        TransformerEncoderUtilizer(
            token_dim=768,
            hidden_dim=128,
            output_dim=768,
            transformer_layers=1,
            transformer_heads=8,
        ),
        CrossAttentionLatentBottleneckUtilizer(
            token_dim=768,
            latent_dim=128,
            hidden_dim=128,
            output_dim=768,
            num_latents=4,
            cross_attention_layers=1,
            num_heads=8,
        ),
        PerceiverLikeUtilizer(
            token_dim=768,
            latent_dim=128,
            hidden_dim=128,
            output_dim=768,
            num_latents=4,
            num_heads=8,
        ),
    ],
)
def test_downstream_utilizer_forward_shape_backward_and_finite(module):
    tokens = torch.randn(2, 16, 768)
    scores = torch.randn(2, 16)
    pred = module(tokens, selected_scores=scores)
    assert pred.shape == (2, 768)
    assert torch.isfinite(pred).all()
    loss = pred.pow(2).mean()
    loss.backward()
    grads = [param.grad for param in module.parameters() if param.requires_grad]
    assert grads
    assert all(grad is None or torch.isfinite(grad).all() for grad in grads)


def test_selector_score_weighted_pool_requires_scores():
    module = SelectorScoreWeightedPoolUtilizer(token_dim=768, hidden_dim=128, output_dim=768)
    with pytest.raises(ValueError, match="selected_scores"):
        module(torch.randn(2, 16, 768))
