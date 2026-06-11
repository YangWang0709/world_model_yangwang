from __future__ import annotations

import torch

from models.teacher_world_model import TeacherWorldModel
from scripts.generate_predictive_importance import compute_occlusion_importance


def test_real_videomae_importance_shapes_with_784_tokens():
    torch.manual_seed(7)
    model = TeacherWorldModel(
        token_dim=768,
        hidden_dim=16,
        output_dim=768,
        num_layers=1,
        dropout=0.0,
        pool="mean",
    )
    past_tokens = torch.randn(1, 784, 768)
    future_tokens = torch.randn(1, 784, 768)

    result = compute_occlusion_importance(
        model,
        past_tokens,
        future_tokens,
        token_chunk_size=16,
        mask_mode="zero",
        mask_value=0.0,
    )

    assert result["importance_scores"].shape == (1, 784)
    assert result["importance_scores_norm"].shape == (1, 784)
    assert result["base_losses"].shape == (1,)
    assert result["masked_losses"].shape == (1, 784)
    for tensor in result.values():
        assert torch.isfinite(tensor).all()
