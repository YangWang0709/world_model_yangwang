"""Current-conditioned selector heads for Step41A diagnosis.

These heads score existing frozen context token artifacts. They do not load or
own VideoMAE, do not consume action/language/future inputs, and are intended
only for bounded current-conditioning smoke diagnostics.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


VARIANT_NAMES = (
    "no_current_context_only",
    "current_mean_summary",
    "current_frame_summary_attention",
    "current_coarse_spatial_query_attention",
    "current_context_similarity_features",
)


class CurrentConditionedSelectorHead(nn.Module):
    """Small Step41A selector head with configurable current-conditioning."""

    def __init__(
        self,
        *,
        variant_name: str,
        token_dim: int = 768,
        hidden_dim: int = 128,
        attention_dim: int = 128,
        context_frames: int = 16,
        current_frames: int = 4,
        spatial_tokens: int = 392,
        coarse_current_bins: int = 49,
        attention_temperature: float = 1.0,
        chunk_context_tokens: int = 784,
        dropout: float = 0.0,
        detach_token_inputs: bool = True,
        use_temporal_embedding: bool = True,
        use_spatial_embedding: bool = True,
    ) -> None:
        super().__init__()
        if variant_name not in VARIANT_NAMES:
            raise ValueError(f"unknown Step41A current-conditioning variant: {variant_name}")
        self.variant_name = str(variant_name)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)
        self.context_frames = int(context_frames)
        self.current_frames = int(current_frames)
        self.spatial_tokens = int(spatial_tokens)
        self.coarse_current_bins = int(coarse_current_bins)
        self.attention_temperature = float(attention_temperature)
        self.chunk_context_tokens = int(chunk_context_tokens)
        self.detach_token_inputs = bool(detach_token_inputs)
        self.use_temporal_embedding = bool(use_temporal_embedding)
        self.use_spatial_embedding = bool(use_spatial_embedding)
        if self.coarse_current_bins <= 0:
            raise ValueError("coarse_current_bins must be positive")
        if self.chunk_context_tokens <= 0:
            raise ValueError("chunk_context_tokens must be positive")

        self.current_mean_proj = nn.Linear(self.token_dim, self.token_dim)
        self.frame_q = nn.Linear(self.token_dim, self.attention_dim)
        self.frame_k = nn.Linear(self.token_dim, self.attention_dim)
        self.frame_v = nn.Linear(self.token_dim, self.token_dim)
        self.coarse_q = nn.Linear(self.token_dim, self.attention_dim)
        self.coarse_k = nn.Linear(self.token_dim, self.attention_dim)
        self.coarse_v = nn.Linear(self.token_dim, self.token_dim)
        self.similarity_proj = nn.Linear(3, self.token_dim)
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

    def forward(self, context_tokens: torch.Tensor, current_tokens: torch.Tensor | None = None) -> dict[str, Any]:
        context = self._context_4d(context_tokens)
        current = None if current_tokens is None else self._current_4d(current_tokens)
        if self.detach_token_inputs:
            context = context.detach()
            current = None if current is None else current.detach()
        x = context.to(dtype=torch.float32)
        current_float = None if current is None else current.to(dtype=torch.float32)

        diagnostics: dict[str, Any] = {
            "coarse_current_bins": self.coarse_current_bins,
            "chunk_context_tokens": self.chunk_context_tokens,
            "token_index_coarse_heuristic": self.variant_name == "current_coarse_spatial_query_attention",
        }
        if self.variant_name == "no_current_context_only":
            conditioned = x
            diagnostics["uses_current_tokens"] = False
        elif self.variant_name == "current_mean_summary":
            conditioned = x + self._current_mean_feature(current_float)
            diagnostics["uses_current_tokens"] = True
        elif self.variant_name == "current_frame_summary_attention":
            conditioned = x + self._frame_attention_feature(x, current_float)
            diagnostics["uses_current_tokens"] = True
        elif self.variant_name == "current_coarse_spatial_query_attention":
            feature, max_elements = self._coarse_spatial_attention_feature(x, current_float)
            conditioned = x + feature
            diagnostics["uses_current_tokens"] = True
            diagnostics["max_attention_elements_seen"] = int(max_elements)
        elif self.variant_name == "current_context_similarity_features":
            conditioned = x + self._similarity_feature(x, current_float)
            diagnostics["uses_current_tokens"] = True
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"unknown variant: {self.variant_name}")

        if self.temporal_embedding is not None:
            conditioned = conditioned + self.temporal_embedding
        if self.spatial_embedding is not None:
            conditioned = conditioned + self.spatial_embedding
        scores = self.score_net(conditioned).squeeze(-1)
        expected = [context.shape[0], self.context_frames, self.spatial_tokens]
        if list(scores.shape) != expected:
            raise ValueError(f"selector scores shape {list(scores.shape)} != expected {expected}")
        return {"scores": scores, "variant_name": self.variant_name, "diagnostics": diagnostics}

    def _current_mean_feature(self, current: torch.Tensor | None) -> torch.Tensor:
        if current is None:
            raise ValueError(f"{self.variant_name} requires current_tokens")
        summary = current.mean(dim=(1, 2))
        return self.current_mean_proj(summary).view(summary.shape[0], 1, 1, self.token_dim)

    def _frame_attention_feature(self, context: torch.Tensor, current: torch.Tensor | None) -> torch.Tensor:
        if current is None:
            raise ValueError(f"{self.variant_name} requires current_tokens")
        context_frame = context.mean(dim=2)
        current_frame = current.mean(dim=2)
        q = self.frame_q(context_frame)
        k = self.frame_k(current_frame)
        v = self.frame_v(current_frame)
        attn = torch.matmul(q, k.transpose(1, 2)) / self._temperature_scale()
        feature = torch.matmul(torch.softmax(attn, dim=-1), v)
        return feature.view(feature.shape[0], self.context_frames, 1, self.token_dim)

    def _coarse_spatial_attention_feature(
        self, context: torch.Tensor, current: torch.Tensor | None
    ) -> tuple[torch.Tensor, int]:
        if current is None:
            raise ValueError(f"{self.variant_name} requires current_tokens")
        batch = context.shape[0]
        prototypes = self._coarse_current_prototypes(current, keep_frames=True)
        k = self.coarse_k(prototypes)
        v = self.coarse_v(prototypes)
        flat_context = context.reshape(batch, self.context_frames * self.spatial_tokens, self.token_dim)
        chunks = []
        max_elements = 0
        for start in range(0, flat_context.shape[1], self.chunk_context_tokens):
            chunk = flat_context[:, start : start + self.chunk_context_tokens]
            q = self.coarse_q(chunk)
            max_elements = max(max_elements, int(q.shape[0] * q.shape[1] * k.shape[1]))
            attn = torch.matmul(q, k.transpose(1, 2)) / self._temperature_scale()
            chunks.append(torch.matmul(torch.softmax(attn, dim=-1), v))
        feature = torch.cat(chunks, dim=1).reshape(batch, self.context_frames, self.spatial_tokens, self.token_dim)
        return feature, max_elements

    def _similarity_feature(self, context: torch.Tensor, current: torch.Tensor | None) -> torch.Tensor:
        if current is None:
            raise ValueError(f"{self.variant_name} requires current_tokens")
        batch = context.shape[0]
        flat_context = context.reshape(batch, self.context_frames * self.spatial_tokens, self.token_dim)
        context_norm = F.normalize(flat_context, dim=-1)
        current_mean = F.normalize(current.mean(dim=(1, 2)), dim=-1)
        prototypes = F.normalize(self._coarse_current_prototypes(current, keep_frames=False), dim=-1)
        mean_sim = (context_norm * current_mean.view(batch, 1, self.token_dim)).sum(dim=-1, keepdim=True)
        proto_sim = torch.matmul(context_norm, prototypes.transpose(1, 2))
        max_sim = proto_sim.max(dim=-1, keepdim=True).values
        avg_sim = proto_sim.mean(dim=-1, keepdim=True)
        sim_features = torch.cat([mean_sim, max_sim, avg_sim], dim=-1)
        projected = self.similarity_proj(sim_features)
        return projected.reshape(batch, self.context_frames, self.spatial_tokens, self.token_dim)

    def _coarse_current_prototypes(self, current: torch.Tensor, *, keep_frames: bool) -> torch.Tensor:
        bins = _contiguous_token_bins(self.spatial_tokens, self.coarse_current_bins, current.device)
        values = []
        for indices in bins:
            values.append(current[:, :, indices, :].mean(dim=2))
        by_frame = torch.stack(values, dim=2)
        if keep_frames:
            return by_frame.reshape(current.shape[0], self.current_frames * len(bins), self.token_dim)
        return by_frame.mean(dim=1)

    def _temperature_scale(self) -> float:
        return max(1.0e-6, math.sqrt(float(self.attention_dim)) * self.attention_temperature)

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
        expected = [self.current_frames, self.spatial_tokens, self.token_dim]
        if list(current_tokens.shape[1:]) != expected:
            raise ValueError(f"current_tokens must be [B, {expected}], got {tuple(current_tokens.shape)}")
        return current_tokens


def build_current_conditioned_selector(config: dict[str, Any], variant_name: str) -> CurrentConditionedSelectorHead:
    model_cfg = config.get("model", {})
    variant_cfg = config.get("current_conditioning_variants", {})
    return CurrentConditionedSelectorHead(
        variant_name=variant_name,
        token_dim=int(model_cfg.get("token_dim", config.get("data", {}).get("token_dim", 768))),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        attention_dim=int(variant_cfg.get("attention_dim", 128)),
        context_frames=int(model_cfg.get("context_frames", config.get("data", {}).get("context_frames", 16))),
        current_frames=int(model_cfg.get("current_frames", config.get("data", {}).get("current_frames", 4))),
        spatial_tokens=int(model_cfg.get("spatial_tokens", config.get("data", {}).get("spatial_tokens", 392))),
        coarse_current_bins=int(variant_cfg.get("coarse_current_bins", 49)),
        attention_temperature=float(variant_cfg.get("attention_temperature", 1.0)),
        chunk_context_tokens=int(variant_cfg.get("chunk_context_tokens", 784)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        detach_token_inputs=bool(variant_cfg.get("detach_token_inputs", True)),
        use_temporal_embedding=bool(variant_cfg.get("use_temporal_embedding", True)),
        use_spatial_embedding=bool(variant_cfg.get("use_spatial_embedding", True)),
    )


def _contiguous_token_bins(spatial_tokens: int, bins: int, device: torch.device) -> list[torch.Tensor]:
    count = min(int(bins), int(spatial_tokens))
    edges = torch.linspace(0, int(spatial_tokens), count + 1, device=device)
    result = []
    for index in range(count):
        start = int(torch.floor(edges[index]).item())
        end = int(torch.floor(edges[index + 1]).item())
        if end <= start:
            end = min(int(spatial_tokens), start + 1)
        result.append(torch.arange(start, end, device=device, dtype=torch.long))
    return result
