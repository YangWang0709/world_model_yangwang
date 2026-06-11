"""Unit test the TeacherWorldModel step for VideoMAE-sized token tensors."""

from __future__ import annotations

import torch

from models.teacher_world_model import TeacherWorldModel
from training.losses import future_latent_mse
from training.teacher_trainer import target_from_future_tokens


def test_teacher_forward_backward_with_videomae_token_shape() -> None:
    batch_size = 2
    num_tokens = 784
    token_dim = 768
    model = TeacherWorldModel(token_dim=token_dim, hidden_dim=128, output_dim=token_dim, num_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    past_tokens = torch.randn(batch_size, num_tokens, token_dim)
    future_tokens = torch.randn(batch_size, num_tokens, token_dim)

    pred = model(past_tokens)
    target = target_from_future_tokens(future_tokens)
    assert list(pred.shape) == [batch_size, token_dim]
    assert list(target.shape) == [batch_size, token_dim]

    loss = future_latent_mse(pred, target)
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
