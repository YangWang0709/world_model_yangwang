"""Proxy-temporal selector head for Step34 planning dry-runs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ProxyTemporalSelectorHead(nn.Module):
    """Small head that scores context frames from frozen token features.

    The module is intentionally scoped as a Step34 scaffold. It consumes
    existing token artifacts and predicts proxy temporal importance labels, but
    it does not own VideoMAE or any downstream world-model training.
    """

    def __init__(
        self,
        *,
        token_dim: int = 768,
        hidden_dim: int = 128,
        context_frames: int = 16,
        spatial_tokens: int = 392,
        condition_on_current_summary: bool = True,
        dropout: float = 0.0,
        detach_token_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_frames = int(context_frames)
        self.spatial_tokens = int(spatial_tokens)
        self.condition_on_current_summary = bool(condition_on_current_summary)
        self.detach_token_inputs = bool(detach_token_inputs)
        self.current_proj = nn.Linear(self.token_dim, self.token_dim)
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
        temporal_features = context.to(dtype=torch.float32).mean(dim=2)
        if self.condition_on_current_summary:
            if current_tokens is None:
                raise ValueError("current_tokens are required when condition_on_current_summary=true")
            current = current_tokens.detach() if self.detach_token_inputs else current_tokens
            current_summary = self._current_summary(current)
            temporal_features = temporal_features + self.current_proj(current_summary).unsqueeze(1)
        scores = self.score_net(temporal_features).squeeze(-1)
        if list(scores.shape) != [context.shape[0], self.context_frames]:
            raise ValueError(f"selector scores shape {list(scores.shape)} is invalid")
        return scores

    def _context_4d(self, context_tokens: torch.Tensor) -> torch.Tensor:
        if context_tokens.ndim == 3:
            batch, tokens, dim = context_tokens.shape
            expected_tokens = self.context_frames * self.spatial_tokens
            if int(tokens) != expected_tokens:
                raise ValueError(f"context token count {tokens} != expected {expected_tokens}")
            context_tokens = context_tokens.reshape(batch, self.context_frames, self.spatial_tokens, dim)
        if context_tokens.ndim == 4:
            if list(context_tokens.shape[1:]) != [self.context_frames, self.spatial_tokens, self.token_dim]:
                raise ValueError(
                    "context_tokens must be "
                    f"[B, {self.context_frames}, {self.spatial_tokens}, {self.token_dim}], "
                    f"got {tuple(context_tokens.shape)}"
                )
            return context_tokens
        raise ValueError(f"context_tokens must be rank 3 or 4, got rank {context_tokens.ndim}")

    def _current_summary(self, current_tokens: torch.Tensor) -> torch.Tensor:
        if current_tokens.ndim == 3:
            if current_tokens.shape[-1] != self.token_dim:
                raise ValueError(f"current token dim {current_tokens.shape[-1]} != {self.token_dim}")
            return current_tokens.to(dtype=torch.float32).mean(dim=1)
        if current_tokens.ndim == 4:
            if current_tokens.shape[-1] != self.token_dim:
                raise ValueError(f"current token dim {current_tokens.shape[-1]} != {self.token_dim}")
            return current_tokens.to(dtype=torch.float32).mean(dim=(1, 2))
        raise ValueError(f"current_tokens must be rank 3 or 4, got rank {current_tokens.ndim}")


def proxy_temporal_targets(context_importance: torch.Tensor) -> torch.Tensor:
    """Convert [B, T, S] or [T, S] proxy importance into [B, T] labels."""

    values = context_importance.detach().to(dtype=torch.float32)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.ndim != 3:
        raise ValueError(f"context_importance must be [B, T, S] or [T, S], got {tuple(values.shape)}")
    target = values.mean(dim=-1)
    row_min = target.amin(dim=1, keepdim=True)
    row_max = target.amax(dim=1, keepdim=True)
    denom = torch.clamp(row_max - row_min, min=1.0e-8)
    return (target - row_min) / denom


def proxy_temporal_selector_dry_run(
    *,
    selector: ProxyTemporalSelectorHead,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run a no-optimizer Step34 forward pass on existing token artifacts."""

    rows: list[dict[str, Any]] = []
    for sample in samples:
        context = _with_batch(sample["context_tokens"])
        current = _with_batch(sample["current_tokens"])
        target = proxy_temporal_targets(_with_batch(sample["context_importance"]))
        with torch.no_grad():
            scores = selector(context, current)
            loss = F.mse_loss(torch.sigmoid(scores), target, reduction="mean")
            random = torch.rand(
                target.shape,
                generator=_generator_for_sample(str(sample.get("sample_id", ""))),
                dtype=target.dtype,
                device=target.device,
            )
            random_loss = F.mse_loss(random, target, reduction="mean")
            current_only = target.mean(dim=1, keepdim=True).expand_as(target)
            current_only_loss = F.mse_loss(current_only, target, reduction="mean")
        rows.append(
            {
                "sample_id": str(sample.get("sample_id")),
                "trajectory_id": sample.get("trajectory_id"),
                "score_shape": list(scores.shape),
                "target_shape": list(target.shape),
                "selector_proxy_temporal_mse": float(loss.item()),
                "random_baseline_mse": float(random_loss.item()),
                "current_only_baseline_mse": float(current_only_loss.item()),
                "loss_finite": bool(torch.isfinite(loss).item()),
                "context_tokens_detached": True,
                "videomae_frozen": True,
                "optimizer_step_performed": False,
                "selector_training_performed": False,
                "current_importance_training_performed": False,
            }
        )
    return {
        "num_samples": len(rows),
        "rows": rows,
        "all_losses_finite": all(bool(row["loss_finite"]) for row in rows),
        "selector_forward_performed": bool(rows),
        "selector_training_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
        "videomae_frozen": True,
        "current_importance_training_performed": False,
    }


def _with_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim in (2, 3):
        return tensor.unsqueeze(0)
    return tensor


def _generator_for_sample(sample_id: str) -> torch.Generator:
    seed = 42 + sum(ord(char) for char in sample_id) % 100_000
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator
