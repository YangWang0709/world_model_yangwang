"""Minimal student world model."""

from __future__ import annotations

import torch
from torch import nn


class StudentWorldModel(nn.Module):
    """Predict a future latent from compressed latents using mean pooling + MLP."""

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 512, output_dim: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, compressed_latents: torch.Tensor) -> torch.Tensor:
        if compressed_latents.ndim != 3:
            raise ValueError(
                f"Expected compressed_latents shape [B, M, H], got {tuple(compressed_latents.shape)}"
            )
        pooled = compressed_latents.mean(dim=1)
        return self.net(pooled)

