"""Minimal task text encoder placeholder."""

from __future__ import annotations

import torch
from torch import nn


class HashTextEncoder(nn.Module):
    """Create deterministic task embeddings without downloading a language model."""

    def __init__(self, embedding_dim: int = 768) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, texts: list[str]) -> torch.Tensor:
        rows = []
        for text in texts:
            seed = sum(ord(ch) for ch in text) % 997
            values = torch.arange(self.embedding_dim, dtype=torch.float32)
            rows.append(torch.sin(values * 0.01 + float(seed)))
        return torch.stack(rows, dim=0)

