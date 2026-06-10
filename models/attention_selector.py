"""Task-conditioned token scoring and top-k selection."""

from __future__ import annotations

import torch
from torch import nn


class AttentionSelector(nn.Module):
    """Score tokens, optionally conditioned on a task embedding."""

    def __init__(
        self,
        token_dim: int = 768,
        hidden_dim: int = 256,
        task_dim: int | None = None,
        use_task: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim or token_dim
        self.use_task = use_task
        self.dropout = dropout

        self.task_proj = nn.Linear(self.task_dim, token_dim)
        layers: list[nn.Module] = [
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.score_net = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor, task_embedding: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens shape [B, N, D], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(f"Expected token dim {self.token_dim}, got {tokens.shape[-1]}")

        conditioned = tokens
        if self.use_task or task_embedding is not None:
            if task_embedding is None:
                raise ValueError("task_embedding is required when use_task=True")
            if task_embedding.ndim != 2 or task_embedding.shape != (tokens.shape[0], self.task_dim):
                raise ValueError(
                    f"Expected task_embedding shape [B, {self.task_dim}], got {tuple(task_embedding.shape)}"
                )
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
