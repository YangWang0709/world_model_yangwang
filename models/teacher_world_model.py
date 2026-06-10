"""Minimal full-token teacher world model."""

from __future__ import annotations

import torch
from torch import nn


class TeacherWorldModel(nn.Module):
    """Predict a future latent from all visual tokens using mean pooling + MLP."""

    def __init__(
        self,
        token_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.0,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        if pool != "mean":
            raise ValueError(f"Unsupported pool {pool!r}; Step 4 supports only 'mean'")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool = pool

        layers: list[nn.Module] = [nn.LayerNorm(token_dim)]
        in_dim = token_dim
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens shape [B, N, D], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(f"Expected token dim {self.token_dim}, got {tokens.shape[-1]}")
        pooled = tokens.mean(dim=1)
        return self.net(pooled)
