"""Small trained predictor teacher for Step30B BridgeData occlusion labels."""

from __future__ import annotations

from typing import Any

import torch


CONTEXT_FRAMES = 16
TOKENS_PER_FRAME = 392
TOKEN_DIM = 768
NUM_CONTEXT_TOKENS = CONTEXT_FRAMES * TOKENS_PER_FRAME


class CurrentConditionedContextAttentionPredictor(torch.nn.Module):
    """Current-conditioned attention pooling predictor used only as a small teacher."""

    def __init__(self, token_dim: int = TOKEN_DIM, hidden_dim: int = 512, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_score_vector = torch.nn.Parameter(torch.empty(self.token_dim))
        self.current_score_vector = torch.nn.Parameter(torch.empty(self.token_dim))
        torch.nn.init.normal_(self.context_score_vector, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.current_score_vector, mean=0.0, std=0.02)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.token_dim * 2, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.token_dim),
        )

    def summarize(
        self,
        context_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context = self._validate_context(context_tokens)
        current = self._validate_current(current_tokens)
        batch = int(context.shape[0])
        context_flat = context.reshape(batch, NUM_CONTEXT_TOKENS, self.token_dim)
        current_summary = current.mean(dim=(1, 2))
        token_scores = torch.matmul(context_flat, self.context_score_vector)
        current_bias = torch.matmul(current_summary, self.current_score_vector).unsqueeze(1)
        attention_logits = token_scores + current_bias
        attention_weights = torch.softmax(attention_logits, dim=1)
        context_summary = torch.bmm(attention_weights.unsqueeze(1), context_flat).squeeze(1)
        return {
            "current_summary": current_summary,
            "context_summary": context_summary,
            "attention_weights": attention_weights,
            "context_flat": context_flat,
        }

    def forward(
        self,
        context_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        summary = self.summarize(context_tokens, current_tokens)
        pred = self.forward_from_summaries(summary["current_summary"], summary["context_summary"])
        if return_attention:
            return pred, summary["attention_weights"]
        return pred

    def forward_from_summaries(self, current_summary: torch.Tensor, context_summary: torch.Tensor) -> torch.Tensor:
        features = torch.cat([current_summary.to(dtype=torch.float32), context_summary.to(dtype=torch.float32)], dim=-1)
        return self.net(features)

    def _validate_context(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise ValueError("context_tokens must be a tensor")
        if value.ndim != 4 or list(value.shape[1:]) != [CONTEXT_FRAMES, TOKENS_PER_FRAME, self.token_dim]:
            raise ValueError(f"context_tokens shape {list(value.shape)} is invalid")
        return value.to(dtype=torch.float32)

    def _validate_current(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise ValueError("current_tokens must be a tensor")
        if value.ndim != 4 or list(value.shape[1:]) != [4, TOKENS_PER_FRAME, self.token_dim]:
            raise ValueError(f"current_tokens shape {list(value.shape)} is invalid")
        return value.to(dtype=torch.float32)


def future_token_summary(future_tokens: torch.Tensor) -> torch.Tensor:
    if not isinstance(future_tokens, torch.Tensor):
        raise ValueError("future_tokens must be a tensor")
    if future_tokens.ndim != 4 or list(future_tokens.shape[1:]) != [4, TOKENS_PER_FRAME, TOKEN_DIM]:
        raise ValueError(f"future_tokens shape {list(future_tokens.shape)} is invalid")
    return future_tokens.to(dtype=torch.float32).mean(dim=(1, 2))


def optimizer_scope_for_teacher(model: CurrentConditionedContextAttentionPredictor, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    model_param_ids = {id(param) for param in model.parameters()}
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group.get("params", [])
        if isinstance(param, torch.nn.Parameter)
    }
    return {
        "optimizer_step_scope": "small_predictor_teacher_only"
        if not (optimizer_param_ids - model_param_ids)
        else "external_parameters_present",
        "optimizer_param_count": len(optimizer_param_ids),
    }
