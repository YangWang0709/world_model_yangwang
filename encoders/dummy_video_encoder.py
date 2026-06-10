"""Deterministic dummy video encoder used by Step 2 smoke tests."""

from __future__ import annotations

import torch
from torch import nn


class DummyVideoEncoder(nn.Module):
    """Map a video batch to dense token-shaped tensors without external weights."""

    def __init__(self, num_tokens: int = 196, token_dim: int = 768) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        basis = torch.linspace(-1.0, 1.0, steps=num_tokens * token_dim)
        self.register_buffer("basis", basis.view(1, num_tokens, token_dim), persistent=False)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [B, T, C, H, W], got {tuple(video.shape)}")

        batch_size = video.shape[0]
        basis = self.basis.to(device=video.device, dtype=video.dtype)
        summary = video.mean(dim=(1, 2, 3, 4), keepdim=False).view(batch_size, 1, 1)
        tokens = basis.expand(batch_size, -1, -1) + summary
        return tokens.contiguous()

