"""Proxy patch/token selector head for Step36 bounded smoke training."""

from __future__ import annotations

import torch
from torch import nn


class ProxyPatchTokenSelectorHead(nn.Module):
    """Small per-token head that scores frozen context token artifacts.

    Step36 trains this head only as a bounded proxy-supervised smoke. The module
    does not own or load VideoMAE, does not consume action/language inputs, and
    detaches token features before applying the trainable selector layers.
    """

    def __init__(
        self,
        *,
        token_dim: int = 768,
        hidden_dim: int = 128,
        context_frames: int = 16,
        spatial_tokens: int = 392,
        condition_on_current_summary: bool = True,
        use_temporal_embedding: bool = True,
        use_spatial_embedding: bool = True,
        dropout: float = 0.0,
        detach_token_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_frames = int(context_frames)
        self.spatial_tokens = int(spatial_tokens)
        self.condition_on_current_summary = bool(condition_on_current_summary)
        self.use_temporal_embedding = bool(use_temporal_embedding)
        self.use_spatial_embedding = bool(use_spatial_embedding)
        self.detach_token_inputs = bool(detach_token_inputs)
        self.current_proj = nn.Linear(self.token_dim, self.token_dim)
        self.temporal_embedding = (
            nn.Parameter(torch.zeros(1, self.context_frames, 1, self.token_dim))
            if self.use_temporal_embedding
            else None
        )
        self.spatial_embedding = (
            nn.Parameter(torch.zeros(1, 1, self.spatial_tokens, self.token_dim))
            if self.use_spatial_embedding
            else None
        )
        layers: list[nn.Module] = [
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.hidden_dim),
            nn.GELU(),
        ]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(self.hidden_dim, 1))
        self.score_net = nn.Sequential(*layers)

    def forward(self, context_tokens: torch.Tensor, current_tokens: torch.Tensor | None = None) -> torch.Tensor:
        context = self._context_4d(context_tokens)
        if self.detach_token_inputs:
            context = context.detach()
        x = context.to(dtype=torch.float32)
        if self.condition_on_current_summary:
            if current_tokens is None:
                raise ValueError("current_tokens are required when condition_on_current_summary=true")
            current = self._current_4d(current_tokens)
            if self.detach_token_inputs:
                current = current.detach()
            current_summary = current.to(dtype=torch.float32).mean(dim=(1, 2))
            x = x + self.current_proj(current_summary).view(current_summary.shape[0], 1, 1, self.token_dim)
        if self.temporal_embedding is not None:
            x = x + self.temporal_embedding
        if self.spatial_embedding is not None:
            x = x + self.spatial_embedding
        scores = self.score_net(x).squeeze(-1)
        expected = [context.shape[0], self.context_frames, self.spatial_tokens]
        if list(scores.shape) != expected:
            raise ValueError(f"selector scores shape {list(scores.shape)} != expected {expected}")
        return scores

    def _context_4d(self, context_tokens: torch.Tensor) -> torch.Tensor:
        if context_tokens.ndim == 3:
            batch, tokens, dim = context_tokens.shape
            expected_tokens = self.context_frames * self.spatial_tokens
            if int(tokens) != expected_tokens:
                raise ValueError(f"context token count {tokens} != expected {expected_tokens}")
            context_tokens = context_tokens.reshape(batch, self.context_frames, self.spatial_tokens, dim)
        if context_tokens.ndim != 4:
            raise ValueError(f"context_tokens must be rank 3 or 4, got rank {context_tokens.ndim}")
        expected = [self.context_frames, self.spatial_tokens, self.token_dim]
        if list(context_tokens.shape[1:]) != expected:
            raise ValueError(f"context_tokens must be [B, {expected}], got {tuple(context_tokens.shape)}")
        return context_tokens

    def _current_4d(self, current_tokens: torch.Tensor) -> torch.Tensor:
        if current_tokens.ndim == 3:
            batch, tokens, dim = current_tokens.shape
            if int(dim) != self.token_dim:
                raise ValueError(f"current token dim {dim} != {self.token_dim}")
            if int(tokens) % self.spatial_tokens != 0:
                raise ValueError(f"current token count {tokens} is not divisible by {self.spatial_tokens}")
            current_tokens = current_tokens.reshape(batch, tokens // self.spatial_tokens, self.spatial_tokens, dim)
        if current_tokens.ndim != 4:
            raise ValueError(f"current_tokens must be rank 3 or 4, got rank {current_tokens.ndim}")
        if int(current_tokens.shape[2]) != self.spatial_tokens or int(current_tokens.shape[3]) != self.token_dim:
            raise ValueError(
                "current_tokens must be [B, Tcur, "
                f"{self.spatial_tokens}, {self.token_dim}], got {tuple(current_tokens.shape)}"
            )
        return current_tokens


def proxy_patch_targets(context_importance: torch.Tensor) -> torch.Tensor:
    """Return detached proxy patch labels as [B, T, S] float32 values."""

    values = context_importance.detach().to(dtype=torch.float32)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.ndim != 3:
        raise ValueError(f"context_importance must be [B, T, S] or [T, S], got {tuple(values.shape)}")
    return values.contiguous()
