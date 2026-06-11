"""Unified predictive-importance selector with a Step17 context mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.attention_selector import AttentionSelector
from training.student_selector_trainer import load_student_selector_checkpoint


class UnifiedPredictiveImportanceSelector(nn.Module):
    """Score either legacy state tokens or historical context tokens.

    Step17 trains only `mode="context"`. `state_legacy` is kept for checkpoint
    compatibility with the earlier Student selector interface.
    """

    def __init__(
        self,
        token_dim: int = 768,
        hidden_dim: int = 256,
        task_dim: int | None = None,
        dropout: float = 0.0,
        condition_on_current: bool = True,
        role_embedding: bool = True,
        temporal_position_embedding: bool = True,
        max_context_tokens: int = 2048,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.task_dim = int(task_dim or token_dim)
        self.condition_on_current = bool(condition_on_current)
        self.role_embedding_enabled = bool(role_embedding)
        self.temporal_position_embedding_enabled = bool(temporal_position_embedding)
        self.max_context_tokens = int(max_context_tokens)
        self.context_selector = AttentionSelector(
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            task_dim=self.task_dim,
            use_task=self.condition_on_current,
            dropout=dropout,
        )
        self.state_legacy_selector = AttentionSelector(
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            task_dim=self.task_dim,
            use_task=False,
            dropout=dropout,
        )
        self.context_role = nn.Parameter(torch.zeros(1, 1, token_dim)) if role_embedding else None
        self.context_pos = (
            nn.Parameter(torch.randn(max_context_tokens, token_dim) * 0.01)
            if temporal_position_embedding
            else None
        )

    def current_summary(self, current_tokens: torch.Tensor | None = None, current_summary: torch.Tensor | None = None) -> torch.Tensor:
        if current_summary is not None:
            if current_summary.ndim != 2 or current_summary.shape[-1] != self.task_dim:
                raise ValueError(f"current_summary must be [B, {self.task_dim}], got {tuple(current_summary.shape)}")
            return current_summary
        if current_tokens is None:
            raise ValueError("current_tokens or current_summary is required for context mode")
        if current_tokens.ndim != 3 or current_tokens.shape[-1] != self.token_dim:
            raise ValueError(f"current_tokens must be [B, N_cur, {self.token_dim}], got {tuple(current_tokens.shape)}")
        return current_tokens.mean(dim=1)

    def forward_context(
        self,
        context_tokens: torch.Tensor,
        current_tokens: torch.Tensor | None = None,
        current_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_tokens.ndim != 3 or context_tokens.shape[-1] != self.token_dim:
            raise ValueError(f"context_tokens must be [B, N_ctx, {self.token_dim}], got {tuple(context_tokens.shape)}")
        hidden = context_tokens
        if self.context_role is not None:
            hidden = hidden + self.context_role
        if self.context_pos is not None:
            if context_tokens.shape[1] > self.max_context_tokens:
                raise ValueError(f"N_ctx={context_tokens.shape[1]} exceeds max_context_tokens={self.max_context_tokens}")
            hidden = hidden + self.context_pos[: context_tokens.shape[1]].unsqueeze(0)
        task = self.current_summary(current_tokens=current_tokens, current_summary=current_summary)
        return self.context_selector(hidden, task if self.condition_on_current else None)

    def forward_state_legacy(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.state_legacy_selector(tokens)

    def forward(
        self,
        tokens: torch.Tensor | None = None,
        *,
        context_tokens: torch.Tensor | None = None,
        current_tokens: torch.Tensor | None = None,
        current_summary: torch.Tensor | None = None,
        mode: str = "context",
    ) -> torch.Tensor:
        if mode == "context":
            context = context_tokens if context_tokens is not None else tokens
            if context is None:
                raise ValueError("context mode requires context_tokens")
            return self.forward_context(context, current_tokens=current_tokens, current_summary=current_summary)
        if mode == "state_legacy":
            legacy = tokens if tokens is not None else context_tokens
            if legacy is None:
                raise ValueError("state_legacy mode requires tokens")
            return self.forward_state_legacy(legacy)
        raise ValueError(f"Unsupported selector mode: {mode!r}")


def build_unified_selector_from_config(config: dict[str, Any]) -> UnifiedPredictiveImportanceSelector:
    return UnifiedPredictiveImportanceSelector(
        token_dim=int(config.get("token_dim", 768)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        task_dim=config.get("task_dim"),
        dropout=float(config.get("dropout", 0.0)),
        condition_on_current=bool(config.get("condition_on_current", True)),
        role_embedding=bool(config.get("role_embedding", True)),
        temporal_position_embedding=bool(config.get("temporal_position_embedding", True)),
        max_context_tokens=int(config.get("max_context_tokens", 2048)),
    )


def initialize_context_selector_from_state_checkpoint(
    selector: UnifiedPredictiveImportanceSelector,
    checkpoint_path: str | Path,
    allow_random_init_if_incompatible: bool = True,
) -> bool:
    """Copy compatible AttentionSelector weights into context/state branches."""

    try:
        checkpoint = load_student_selector_checkpoint(checkpoint_path, map_location="cpu")
        legacy = AttentionSelector(**checkpoint["model_config"])
        legacy.load_state_dict(checkpoint["model_state_dict"])
        selector.state_legacy_selector.load_state_dict(legacy.state_dict(), strict=False)
        context_state = selector.context_selector.state_dict()
        compatible = {
            key: value
            for key, value in legacy.state_dict().items()
            if key in context_state and tuple(context_state[key].shape) == tuple(value.shape)
        }
        selector.context_selector.load_state_dict({**context_state, **compatible}, strict=False)
        return bool(compatible)
    except Exception:
        if allow_random_init_if_incompatible:
            return False
        raise
