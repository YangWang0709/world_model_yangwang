"""Small diagnostic predictor variants for Step31 BridgeData analysis."""

from __future__ import annotations

from typing import Any

import torch

TOKEN_DIM = 768
CONTEXT_FRAMES = 16
TOKENS_PER_FRAME = 392
DIAGNOSTIC_OPTIMIZER_SCOPE = "diagnostic_predictor_only"

DIAGNOSTIC_VARIANTS = (
    "current_only_predictor",
    "current_plus_mean_context_predictor",
    "current_plus_proxy_topk_context_predictor",
    "current_conditioned_attention_predictor",
    "current_conditioned_attention_with_proxy_prior",
    "temporal_frame_attention_predictor",
)


class BridgeDataTeacherDiagnosticPredictor(torch.nn.Module):
    """Tiny MLP that predicts a future summary from current/context summaries."""

    def __init__(self, variant: str, token_dim: int = TOKEN_DIM, hidden_dim: int = 256, topk: int = 256, seed: int = 42) -> None:
        super().__init__()
        if variant not in DIAGNOSTIC_VARIANTS:
            raise ValueError(f"unknown diagnostic variant: {variant}")
        torch.manual_seed(int(seed))
        self.variant = str(variant)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.topk = int(topk)
        self.score_vector = torch.nn.Parameter(torch.empty(self.token_dim))
        torch.nn.init.normal_(self.score_vector, mean=0.0, std=0.02)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.token_dim * 2, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.token_dim),
        )

    def forward(
        self,
        context_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        context_importance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current_summary, context_summary = diagnostic_summaries(
            context_tokens=context_tokens,
            current_tokens=current_tokens,
            context_importance=context_importance,
            variant=self.variant,
            topk=self.topk,
            score_vector=self.score_vector,
        )
        return self.forward_from_summaries(current_summary, context_summary)

    def forward_from_summaries(self, current_summary: torch.Tensor, context_summary: torch.Tensor) -> torch.Tensor:
        features = torch.cat([current_summary.to(dtype=torch.float32), context_summary.to(dtype=torch.float32)], dim=-1)
        return self.net(features)


def diagnostic_summaries(
    context_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    context_importance: torch.Tensor | None,
    variant: str,
    topk: int = 256,
    score_vector: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context_tokens.ndim != 4 or list(context_tokens.shape[1:]) != [CONTEXT_FRAMES, TOKENS_PER_FRAME, TOKEN_DIM]:
        raise ValueError(f"context_tokens shape is invalid: {list(context_tokens.shape)}")
    if current_tokens.ndim != 4 or list(current_tokens.shape[1:]) != [4, TOKENS_PER_FRAME, TOKEN_DIM]:
        raise ValueError(f"current_tokens shape is invalid: {list(current_tokens.shape)}")
    if variant not in DIAGNOSTIC_VARIANTS:
        raise ValueError(f"unknown diagnostic variant: {variant}")
    context = context_tokens.to(dtype=torch.float32)
    current = current_tokens.to(dtype=torch.float32)
    batch = int(context.shape[0])
    current_summary = current.mean(dim=(1, 2))
    context_flat = context.reshape(batch, CONTEXT_FRAMES * TOKENS_PER_FRAME, TOKEN_DIM)

    if variant == "current_only_predictor":
        return current_summary, torch.zeros_like(current_summary)
    if variant == "current_plus_mean_context_predictor":
        return current_summary, context_flat.mean(dim=1)
    if variant == "current_plus_proxy_topk_context_predictor":
        return current_summary, _topk_context_summary(context_flat, _importance_flat(context_importance, batch), topk)
    if variant == "current_conditioned_attention_with_proxy_prior":
        scores = _attention_scores(context_flat, score_vector)
        if context_importance is not None:
            scores = scores + _importance_flat(context_importance, batch)
        return current_summary, _weighted_summary(context_flat, scores)
    if variant == "temporal_frame_attention_predictor":
        frame_summary = context.mean(dim=2)
        scores = torch.matmul(frame_summary, score_vector.to(context.device, dtype=torch.float32)) if score_vector is not None else frame_summary.mean(dim=-1)
        weights = torch.softmax(scores, dim=1)
        return current_summary, torch.bmm(weights.unsqueeze(1), frame_summary).squeeze(1)
    return current_summary, _weighted_summary(context_flat, _attention_scores(context_flat, score_vector))


def target_summary(future_tokens: torch.Tensor, current_tokens: torch.Tensor, target_variant: str = "future_mean_all4") -> torch.Tensor:
    if future_tokens.ndim != 4 or list(future_tokens.shape[1:]) != [4, TOKENS_PER_FRAME, TOKEN_DIM]:
        raise ValueError(f"future_tokens shape is invalid: {list(future_tokens.shape)}")
    future = future_tokens.to(dtype=torch.float32)
    current = current_tokens.to(dtype=torch.float32)
    if target_variant == "future_mean_all4":
        return future.mean(dim=(1, 2))
    if target_variant == "future_first":
        return future[:, 0].mean(dim=1)
    if target_variant == "future_last":
        return future[:, -1].mean(dim=1)
    if target_variant == "future_delta_last_minus_current":
        return future[:, -1].mean(dim=1) - current[:, -1].mean(dim=1)
    raise ValueError(f"unknown target variant: {target_variant}")


def optimizer_scope_for_diagnostic_predictor(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    model_param_ids = {id(param) for param in model.parameters()}
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group.get("params", [])
        if isinstance(param, torch.nn.Parameter)
    }
    return {
        "optimizer_step_scope": DIAGNOSTIC_OPTIMIZER_SCOPE
        if not (optimizer_param_ids - model_param_ids)
        else "external_parameters_present",
        "optimizer_param_count": len(optimizer_param_ids),
    }


def _importance_flat(context_importance: torch.Tensor | None, batch: int) -> torch.Tensor:
    if context_importance is None:
        return torch.zeros((batch, CONTEXT_FRAMES * TOKENS_PER_FRAME), dtype=torch.float32)
    if context_importance.ndim != 3 or list(context_importance.shape[1:]) != [CONTEXT_FRAMES, TOKENS_PER_FRAME]:
        raise ValueError(f"context_importance shape is invalid: {list(context_importance.shape)}")
    return context_importance.to(dtype=torch.float32).reshape(batch, CONTEXT_FRAMES * TOKENS_PER_FRAME)


def _topk_context_summary(context_flat: torch.Tensor, scores: torch.Tensor, topk: int) -> torch.Tensor:
    k = max(1, min(int(topk), int(context_flat.shape[1])))
    index = torch.topk(scores.to(context_flat.device), k=k, dim=1).indices
    gathered = torch.gather(context_flat, 1, index.unsqueeze(-1).expand(-1, -1, context_flat.shape[-1]))
    return gathered.mean(dim=1)


def _attention_scores(context_flat: torch.Tensor, score_vector: torch.Tensor | None) -> torch.Tensor:
    if score_vector is None:
        return context_flat.mean(dim=-1)
    return torch.matmul(context_flat, score_vector.to(context_flat.device, dtype=torch.float32))


def _weighted_summary(context_flat: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(scores.to(context_flat.device, dtype=torch.float32), dim=1)
    return torch.bmm(weights.unsqueeze(1), context_flat).squeeze(1)
