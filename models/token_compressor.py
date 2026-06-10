"""Minimal token compressor with a Perceiver-like interface."""

from __future__ import annotations

import torch
from torch import nn


class TokenCompressor(nn.Module):
    """Compress selected tokens into a fixed number of latent vectors."""

    def __init__(
        self,
        token_dim: int = 768,
        num_latents: int = 16,
        latent_dim: int = 512,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.input_proj = nn.Linear(token_dim, latent_dim)
        self.latent_queries = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, selected_tokens: torch.Tensor) -> torch.Tensor:
        if selected_tokens.ndim != 3:
            raise ValueError(
                f"Expected selected_tokens shape [B, K, D], got {tuple(selected_tokens.shape)}"
            )

        batch_size = selected_tokens.shape[0]
        key_value = self.input_proj(selected_tokens)
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        compressed, _ = self.cross_attn(query=queries, key=key_value, value=key_value, need_weights=False)
        return self.norm(compressed)

