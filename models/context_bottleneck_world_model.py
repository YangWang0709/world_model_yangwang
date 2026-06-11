"""Context teacher and bottleneck world model for Step 17."""

from __future__ import annotations

import torch
from torch import nn


def future_target_from_tokens(future_tokens: torch.Tensor) -> torch.Tensor:
    if future_tokens.ndim == 3:
        return future_tokens.mean(dim=1)
    if future_tokens.ndim == 2:
        return future_tokens
    raise ValueError(f"future_tokens must be [B, N, D] or [B, D], got {tuple(future_tokens.shape)}")


class _ConcatMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ContextTeacherWorldModel(nn.Module):
    """Predict a future latent from full context and full current tokens."""

    def __init__(
        self,
        token_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.0,
        current_pool: str = "mean",
        context_pool: str = "mean",
        fusion: str = "concat_mlp",
    ) -> None:
        super().__init__()
        if current_pool != "mean" or context_pool != "mean" or fusion != "concat_mlp":
            raise ValueError("Step17 context teacher supports mean/mean/concat_mlp only")
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.current_pool = current_pool
        self.context_pool = context_pool
        self.fusion = fusion
        self.head = _ConcatMLP(token_dim * 2, hidden_dim, output_dim, num_layers, dropout)

    def forward(self, context_tokens: torch.Tensor, current_tokens: torch.Tensor) -> torch.Tensor:
        if context_tokens.ndim != 3 or current_tokens.ndim != 3:
            raise ValueError("context_tokens and current_tokens must be [B, N, D]")
        if context_tokens.shape[0] != current_tokens.shape[0]:
            raise ValueError("context/current batch sizes must match")
        if context_tokens.shape[-1] != self.token_dim or current_tokens.shape[-1] != self.token_dim:
            raise ValueError(f"context/current token dim must be {self.token_dim}")
        return self.head(torch.cat([context_tokens.mean(dim=1), current_tokens.mean(dim=1)], dim=-1))


class ContextBottleneckWorldModel(nn.Module):
    """Use full current summary plus compressed selected context memory."""

    def __init__(
        self,
        token_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.0,
        context_memory_mode: str = "mean_pool_selected_context",
    ) -> None:
        super().__init__()
        if context_memory_mode not in {"mean_pool_selected_context", "selector_score_weighted_pool"}:
            raise ValueError(f"Unsupported context_memory_mode: {context_memory_mode!r}")
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.context_memory_mode = context_memory_mode
        self.head = _ConcatMLP(token_dim * 2, hidden_dim, output_dim, num_layers, dropout)

    def context_memory(self, selected_context_tokens: torch.Tensor, selected_scores: torch.Tensor | None = None) -> torch.Tensor:
        if selected_context_tokens.ndim != 3 or selected_context_tokens.shape[-1] != self.token_dim:
            raise ValueError(f"selected_context_tokens must be [B, K, {self.token_dim}]")
        if self.context_memory_mode == "selector_score_weighted_pool":
            if selected_scores is None:
                raise ValueError("selected_scores is required for selector_score_weighted_pool")
            if selected_scores.shape != selected_context_tokens.shape[:2]:
                raise ValueError("selected_scores must be [B, K]")
            weights = torch.softmax(selected_scores.float(), dim=1).unsqueeze(-1)
            return (selected_context_tokens * weights).sum(dim=1)
        return selected_context_tokens.mean(dim=1)

    def forward(
        self,
        current_tokens: torch.Tensor,
        selected_context_tokens: torch.Tensor,
        selected_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if current_tokens.ndim != 3 or current_tokens.shape[-1] != self.token_dim:
            raise ValueError(f"current_tokens must be [B, N_cur, {self.token_dim}]")
        current_summary = current_tokens.mean(dim=1)
        context_memory = self.context_memory(selected_context_tokens, selected_scores)
        return self.head(torch.cat([context_memory, current_summary], dim=-1))
