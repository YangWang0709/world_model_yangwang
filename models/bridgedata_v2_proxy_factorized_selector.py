"""Factorized proxy selector heads for Step37 bounded diagnosis."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ProxyFactorizedSelectorHead(nn.Module):
    """Small temporal-plus-spatial-residual selector for frozen token artifacts."""

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
        use_temporal_branch: bool = True,
        use_spatial_residual_branch: bool = True,
        combine_mode: str = "temporal_plus_spatial_residual",
        detach_token_inputs: bool = True,
    ) -> None:
        super().__init__()
        if combine_mode != "temporal_plus_spatial_residual":
            raise ValueError(f"unsupported combine_mode: {combine_mode!r}")
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_frames = int(context_frames)
        self.spatial_tokens = int(spatial_tokens)
        self.condition_on_current_summary = bool(condition_on_current_summary)
        self.use_temporal_branch = bool(use_temporal_branch)
        self.use_spatial_residual_branch = bool(use_spatial_residual_branch)
        self.detach_token_inputs = bool(detach_token_inputs)
        self.current_proj = nn.Linear(self.token_dim, self.token_dim)
        self.temporal_embedding = (
            nn.Parameter(torch.zeros(1, self.context_frames, 1, self.token_dim))
            if bool(use_temporal_embedding)
            else None
        )
        self.spatial_embedding = (
            nn.Parameter(torch.zeros(1, 1, self.spatial_tokens, self.token_dim))
            if bool(use_spatial_embedding)
            else None
        )
        self.temporal_mlp = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.spatial_mlp = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, context_tokens: torch.Tensor, current_tokens: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
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
        frame_features = x.mean(dim=2)
        if self.use_temporal_branch:
            temporal_scores = self.temporal_mlp(frame_features).squeeze(-1)
        else:
            temporal_scores = x.new_zeros((x.shape[0], self.context_frames))
        if self.use_spatial_residual_branch:
            spatial_residual_scores = self.spatial_mlp(x).squeeze(-1)
        else:
            spatial_residual_scores = x.new_zeros((x.shape[0], self.context_frames, self.spatial_tokens))
        scores = temporal_scores.unsqueeze(-1) + spatial_residual_scores
        expected_scores = [context.shape[0], self.context_frames, self.spatial_tokens]
        if list(scores.shape) != expected_scores:
            raise ValueError(f"scores shape {list(scores.shape)} != expected {expected_scores}")
        if list(temporal_scores.shape) != [context.shape[0], self.context_frames]:
            raise ValueError("temporal_scores shape is invalid")
        return {
            "scores": scores,
            "temporal_scores": temporal_scores,
            "spatial_residual_scores": spatial_residual_scores,
        }

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


class ProxyFactorizedSelectorWithProxyTemporalPrior(ProxyFactorizedSelectorHead):
    """Factorized selector class marker for auxiliary proxy-temporal training."""

    uses_proxy_temporal_auxiliary_target = True


def proxy_patch_and_temporal_targets(context_importance: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return patch target [B, T, S] and normalized temporal target [B, T]."""

    patch = context_importance.detach().to(dtype=torch.float32)
    if patch.ndim == 2:
        patch = patch.unsqueeze(0)
    if patch.ndim != 3:
        raise ValueError(f"context_importance must be [B, T, S] or [T, S], got {tuple(patch.shape)}")
    temporal = patch.mean(dim=-1)
    row_min = temporal.amin(dim=1, keepdim=True)
    row_max = temporal.amax(dim=1, keepdim=True)
    temporal = (temporal - row_min) / torch.clamp(row_max - row_min, min=1.0e-8)
    return {"proxy_patch_target": patch.contiguous(), "proxy_temporal_target": temporal.contiguous()}


def validate_factorized_output(output: dict[str, Any], *, batch: int, frames: int, spatial_tokens: int) -> bool:
    expected_scores = [int(batch), int(frames), int(spatial_tokens)]
    expected_temporal = [int(batch), int(frames)]
    if list(output["scores"].shape) != expected_scores:
        raise ValueError(f"scores shape {list(output['scores'].shape)} != {expected_scores}")
    if list(output["spatial_residual_scores"].shape) != expected_scores:
        raise ValueError("spatial_residual_scores shape is invalid")
    if list(output["temporal_scores"].shape) != expected_temporal:
        raise ValueError("temporal_scores shape is invalid")
    return True
