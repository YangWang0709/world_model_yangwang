"""Minimal full-token teacher world model."""

from __future__ import annotations

import torch
from torch import nn


class TeacherWorldModel(nn.Module):
    """Predict a future latent from all visual tokens using mean pooling + MLP."""

    def __init__(self, token_dim: int = 768, hidden_dim: int = 512, output_dim: int | None = None) -> None:
        super().__init__()
        output_dim = output_dim or token_dim
        self.net = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens shape [B, N, D], got {tuple(tokens.shape)}")
        pooled = tokens.mean(dim=1)
        return self.net(pooled)

