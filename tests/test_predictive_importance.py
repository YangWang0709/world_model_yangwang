from __future__ import annotations

import torch

from models.teacher_world_model import TeacherWorldModel
from scripts.generate_predictive_importance import compute_occlusion_importance


def test_compute_occlusion_importance_shapes_and_finite():
    model = TeacherWorldModel(token_dim=4, hidden_dim=8, output_dim=4, num_layers=2)
    past_tokens = torch.randn(2, 5, 4)
    future_tokens = torch.randn(2, 5, 4)

    result = compute_occlusion_importance(
        model,
        past_tokens,
        future_tokens,
        token_chunk_size=2,
        mask_mode="zero",
        mask_value=0.0,
    )

    assert result["base_losses"].shape == (2,)
    assert result["masked_losses"].shape == (2, 5)
    assert result["importance_scores"].shape == (2, 5)
    assert result["importance_scores_norm"].shape == (2, 5)
    for value in result.values():
        assert torch.isfinite(value).all()
