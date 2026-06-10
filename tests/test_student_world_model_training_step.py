from __future__ import annotations

import torch

from models.student_world_model import StudentWorldModel
from models.token_compressor import TokenCompressor


def test_student_world_model_training_step_backward_and_optimizer() -> None:
    compressor = TokenCompressor(
        token_dim=6,
        latent_dim=8,
        num_latents=3,
        hidden_dim=10,
        num_heads=2,
        dropout=0.0,
    )
    student = StudentWorldModel(latent_dim=8, hidden_dim=12, output_dim=6, num_layers=2, dropout=0.0)
    optimizer = torch.optim.AdamW(list(compressor.parameters()) + list(student.parameters()), lr=1e-3)

    selected_tokens = torch.randn(2, 4, 6)
    target = torch.randn(2, 6)
    compressed = compressor(selected_tokens)
    pred = student(compressed)
    loss = torch.nn.functional.mse_loss(pred, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert compressed.shape == (2, 3, 8)
    assert pred.shape == (2, 6)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(loss)
