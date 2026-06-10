"""Task-conditioned token scoring and top-k selection."""

from __future__ import annotations

import torch
from torch import nn


class AttentionSelector(nn.Module):
    """Score tokens, optionally conditioned on a task embedding."""

    def __init__(self, token_dim: int = 768, hidden_dim: int = 256) -> None:
        super().__init__()
        self.task_proj = nn.Linear(token_dim, token_dim)
        self.score_net = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, task_embedding: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens shape [B, N, D], got {tuple(tokens.shape)}")

        conditioned = tokens
        if task_embedding is not None:
            if task_embedding.ndim != 2 or task_embedding.shape[0] != tokens.shape[0]:
                raise ValueError("Expected task_embedding shape [B, D]")
            conditioned = conditioned + self.task_proj(task_embedding).unsqueeze(1)

        return self.score_net(conditioned).squeeze(-1)


def select_topk(tokens: torch.Tensor, scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the top-k tokens per sample according to scores."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens shape [B, N, D], got {tuple(tokens.shape)}")
    if scores.ndim != 2:
        raise ValueError(f"Expected scores shape [B, N], got {tuple(scores.shape)}")
    if tokens.shape[:2] != scores.shape:
        raise ValueError("Token and score batch/token dimensions must match")
    if k <= 0 or k > tokens.shape[1]:
        raise ValueError(f"k must be in [1, {tokens.shape[1]}], got {k}")

    indices = torch.topk(scores, k=k, dim=1).indices
    gather_index = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    selected = torch.gather(tokens, dim=1, index=gather_index)
    return selected.contiguous(), indices

