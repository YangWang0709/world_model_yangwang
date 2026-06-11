from __future__ import annotations

import torch

from models.student_world_model import StudentWorldModel
from models.token_compressor import TokenCompressor


def test_bair_videomae_student_world_model_training_step_backward_and_optimizer() -> None:
    compressor = TokenCompressor(
        token_dim=768,
        latent_dim=32,
        num_latents=4,
        hidden_dim=64,
        num_heads=4,
        dropout=0.0,
    )
    student = StudentWorldModel(latent_dim=32, hidden_dim=64, output_dim=768, num_layers=2, dropout=0.0)
    optimizer = torch.optim.AdamW(list(compressor.parameters()) + list(student.parameters()), lr=1e-3)

    selected_tokens = torch.randn(2, 16, 768)
    target = torch.randn(2, 768)
    compressed = compressor(selected_tokens)
    pred = student(compressed)
    loss = torch.nn.functional.mse_loss(pred, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert compressed.shape == (2, 4, 32)
    assert pred.shape == (2, 768)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(loss)
