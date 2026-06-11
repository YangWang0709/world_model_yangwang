"""Downstream token-utilization modules for Step 16 ablations."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.student_world_model import StudentWorldModel
from models.token_compressor import TokenCompressor


class BaseDownstreamUtilizer(nn.Module):
    """Predict a future latent from selected tokens."""

    def forward(
        self,
        selected_tokens: torch.Tensor,
        selected_scores: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def _validate_selected_tokens(selected_tokens: torch.Tensor, token_dim: int) -> None:
    if selected_tokens.ndim != 3:
        raise ValueError(f"Expected selected_tokens [B, K, D], got {tuple(selected_tokens.shape)}")
    if selected_tokens.shape[-1] != token_dim:
        raise ValueError(f"Expected token dim {token_dim}, got {selected_tokens.shape[-1]}")


class _PredictionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 768,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class MeanPoolUtilizer(BaseDownstreamUtilizer):
    """Mean-pool selected tokens, then predict the future latent."""

    def __init__(
        self,
        token_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        dropout: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.head = _PredictionHead(self.token_dim, hidden_dim=int(hidden_dim), output_dim=int(output_dim), dropout=float(dropout))

    def forward(
        self,
        selected_tokens: torch.Tensor,
        selected_scores: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del selected_scores, metadata
        _validate_selected_tokens(selected_tokens, self.token_dim)
        return self.head(selected_tokens.mean(dim=1))


class AttentionPoolUtilizer(BaseDownstreamUtilizer):
    """Use one learned query to attend over selected tokens."""

    def __init__(
        self,
        token_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        dropout: float = 0.0,
        num_heads: int = 8,
        **_: Any,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_proj = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim))
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=int(num_heads), dropout=float(dropout), batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = _PredictionHead(hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)

    def forward(
        self,
        selected_tokens: torch.Tensor,
        selected_scores: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del selected_scores, metadata
        _validate_selected_tokens(selected_tokens, self.token_dim)
        keys = self.token_proj(selected_tokens)
        query = self.query.expand(selected_tokens.shape[0], -1, -1)
        pooled, _ = self.attn(query=query, key=keys, value=keys, need_weights=False)
        return self.head(self.norm(pooled.squeeze(1)))


class SelectorScoreWeightedPoolUtilizer(BaseDownstreamUtilizer):
    """Pool selected tokens using deployable selector scores."""

    def __init__(
        self,
        token_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        dropout: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.head = _PredictionHead(self.token_dim, hidden_dim=int(hidden_dim), output_dim=int(output_dim), dropout=float(dropout))

    def forward(
        self,
        selected_tokens: torch.Tensor,
        selected_scores: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del metadata
        _validate_selected_tokens(selected_tokens, self.token_dim)
        if selected_scores is None:
            raise ValueError("selected_scores is required for selector_score_weighted_pool")
        if selected_scores.shape != selected_tokens.shape[:2]:
            raise ValueError(
                f"Expected selected_scores [B, K] matching tokens, got {tuple(selected_scores.shape)}"
            )
        weights = torch.softmax(selected_scores.float(), dim=1).unsqueeze(-1)
        pooled = (selected_tokens * weights).sum(dim=1)
        return self.head(pooled)


class TransformerEncoderUtilizer(BaseDownstreamUtilizer):
    """Run a small TransformerEncoder over selected tokens."""

    def __init__(
        self,
        token_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        dropout: float = 0.0,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        max_tokens: int = 512,
        pool: str = "mean",
        **_: Any,
    ) -> None:
        super().__init__()
        if hidden_dim % transformer_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if pool not in {"mean", "cls"}:
            raise ValueError("pool must be 'mean' or 'cls'")
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_tokens = int(max_tokens)
        self.pool = pool
        self.token_proj = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim))
        self.pos = nn.Parameter(torch.randn(max_tokens, hidden_dim) * 0.01)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02) if pool == "cls" else None
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(transformer_heads),
            dim_feedforward=hidden_dim * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(transformer_layers))
        self.head = _PredictionHead(hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)

    def forward(
        self,
        selected_tokens: torch.Tensor,
        selected_scores: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del selected_scores, metadata
        _validate_selected_tokens(selected_tokens, self.token_dim)
        if selected_tokens.shape[1] > self.max_tokens:
            raise ValueError(f"K={selected_tokens.shape[1]} exceeds max_tokens={self.max_tokens}")
        hidden = self.token_proj(selected_tokens) + self.pos[: selected_tokens.shape[1]].unsqueeze(0)
        if self.cls is not None:
            hidden = torch.cat([self.cls.expand(hidden.shape[0], -1, -1), hidden], dim=1)
        encoded = self.encoder(hidden)
        pooled = encoded[:, 0] if self.pool == "cls" else encoded.mean(dim=1)
        return self.head(pooled)


class _CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, latents: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        cross_out, _ = self.cross(query=self.norm1(latents), key=key_value, value=key_value, need_weights=False)
        latents = latents + cross_out
        self_out, _ = self.self_attn(query=self.norm2(latents), key=self.norm2(latents), value=self.norm2(latents), need_weights=False)
        latents = latents + self_out
        return latents + self.ffn(self.norm3(latents))


class CrossAttentionLatentBottleneckUtilizer(BaseDownstreamUtilizer):
    """Cross-attend learned latents to the selected-token set."""

    def __init__(
        self,
        token_dim: int = 768,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_latents: int = 16,
        cross_attention_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        if hidden_dim != latent_dim:
            hidden_dim = latent_dim
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")
        self.token_dim = int(token_dim)
        self.latent_dim = int(latent_dim)
        self.token_proj = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, latent_dim))
        self.latent_queries = nn.Parameter(torch.randn(int(num_latents), latent_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [_CrossAttentionBlock(latent_dim, int(num_heads), float(dropout)) for _ in range(int(cross_attention_layers))]
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.head = _PredictionHead(latent_dim, hidden_dim=latent_dim, output_dim=output_dim, dropout=dropout)

    def forward(
        self,
        selected_tokens: torch.Tensor,
        selected_scores: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del selected_scores, metadata
        _validate_selected_tokens(selected_tokens, self.token_dim)
        key_value = self.token_proj(selected_tokens)
        latents = self.latent_queries.unsqueeze(0).expand(selected_tokens.shape[0], -1, -1)
        for block in self.blocks:
            latents = block(latents, key_value)
        return self.head(self.norm(latents).mean(dim=1))


class PerceiverLikeUtilizer(BaseDownstreamUtilizer):
    """Wrapper for the existing TokenCompressor + StudentWorldModel baseline."""

    def __init__(
        self,
        token_dim: int = 768,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_latents: int = 16,
        dropout: float = 0.0,
        compressor_type: str = "perceiver_like",
        num_heads: int = 8,
        num_layers: int = 2,
        pool: str = "mean",
        **_: Any,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.compressor = TokenCompressor(
            token_dim=token_dim,
            latent_dim=latent_dim,
            num_latents=num_latents,
            hidden_dim=hidden_dim,
            dropout=dropout,
            compressor_type=compressor_type,
            num_heads=num_heads,
        )
        self.student_world_model = StudentWorldModel(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            pool=pool,
        )

    def forward(
        self,
        selected_tokens: torch.Tensor,
        selected_scores: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del selected_scores, metadata
        _validate_selected_tokens(selected_tokens, self.token_dim)
        return self.student_world_model(self.compressor(selected_tokens))


def build_downstream_utilizer(utilization_module: str, **kwargs: Any) -> BaseDownstreamUtilizer:
    """Build a Step16 utilizer by config name."""

    builders = {
        "mean_pool": MeanPoolUtilizer,
        "attention_pool": AttentionPoolUtilizer,
        "selector_score_weighted_pool": SelectorScoreWeightedPoolUtilizer,
        "transformer_encoder": TransformerEncoderUtilizer,
        "cross_attention_latent_bottleneck": CrossAttentionLatentBottleneckUtilizer,
        "perceiver_like": PerceiverLikeUtilizer,
    }
    if utilization_module not in builders:
        raise ValueError(f"Unknown utilization_module: {utilization_module!r}")
    return builders[utilization_module](**kwargs)
