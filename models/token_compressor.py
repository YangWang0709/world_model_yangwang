"""Minimal token compressor with a Perceiver-like interface."""

from __future__ import annotations

import torch
from torch import nn


class TokenCompressor(nn.Module):
    """Compress selected tokens into a fixed number of latent vectors."""

    def __init__(
        self,
        token_dim: int = 768,
        latent_dim: int = 512,
        num_latents: int = 16,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        compressor_type: str = "perceiver_like",
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if compressor_type not in {"perceiver_like", "mean_mlp"}:
            raise ValueError(f"Unsupported compressor_type {compressor_type!r}")
        if num_latents <= 0:
            raise ValueError("num_latents must be positive")
        if latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim and hidden_dim must be positive")
        if compressor_type == "perceiver_like" and latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads for perceiver_like compression")

        self.token_dim = token_dim
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.compressor_type = compressor_type
        self.num_heads = num_heads

        if hidden_dim == latent_dim and dropout == 0.0:
            self.input_proj = nn.Linear(token_dim, latent_dim)
        else:
            layers: list[nn.Module] = [
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
            ]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, latent_dim))
            self.input_proj = nn.Sequential(*layers)

        self.latent_queries = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)
        self.out_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, selected_tokens: torch.Tensor) -> torch.Tensor:
        if selected_tokens.ndim != 3:
            raise ValueError(
                f"Expected selected_tokens shape [B, K, D], got {tuple(selected_tokens.shape)}"
            )
        if selected_tokens.shape[-1] != self.token_dim:
            raise ValueError(f"Expected token dim {self.token_dim}, got {selected_tokens.shape[-1]}")

        batch_size = selected_tokens.shape[0]
        key_value = self.input_proj(selected_tokens)
        if self.compressor_type == "mean_mlp":
            pooled = key_value.mean(dim=1, keepdim=True)
            return self.norm(pooled.expand(batch_size, self.num_latents, self.latent_dim).contiguous())

        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        compressed, _ = self.cross_attn(query=queries, key=key_value, value=key_value, need_weights=False)
        return self.norm(self.out_dropout(compressed))
