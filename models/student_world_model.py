"""Minimal student world model."""

from __future__ import annotations

import torch
from torch import nn


class StudentWorldModel(nn.Module):
    """Predict a future latent from compressed latents using mean pooling + MLP."""

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.0,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        if pool != "mean":
            raise ValueError(f"Unsupported pool {pool!r}; StudentWorldModel supports only 'mean'")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool = pool

        layers: list[nn.Module] = [nn.LayerNorm(latent_dim)]
        in_dim = latent_dim
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, compressed_latents: torch.Tensor) -> torch.Tensor:
        if compressed_latents.ndim != 3:
            raise ValueError(
                f"Expected compressed_latents shape [B, M, H], got {tuple(compressed_latents.shape)}"
            )
        if compressed_latents.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {compressed_latents.shape[-1]}")
        pooled = compressed_latents.mean(dim=1)
        return self.net(pooled)
