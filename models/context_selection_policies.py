"""Context-only selection policies for Step 17."""

from __future__ import annotations

from typing import Any

import torch

from models.selection_policies import compute_teacher_importance_selection_metrics, gather_tokens_by_indices
from models.unified_predictive_importance_selector import UnifiedPredictiveImportanceSelector


def topk_from_retention(num_tokens: int, retention_ratio: float, fallback_topk: int = 32) -> int:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if retention_ratio <= 0:
        return min(int(fallback_topk), int(num_tokens))
    return max(1, min(int(num_tokens), int(round(float(num_tokens) * float(retention_ratio)))))


def uniform_context_indices(batch_size: int, num_tokens: int, k: int, device: torch.device) -> torch.Tensor:
    base = torch.linspace(0, num_tokens - 1, steps=k).round().long()
    return base.unsqueeze(0).expand(batch_size, -1).to(device)


def hybrid_context_indices(learned_sorted: torch.Tensor, uniform: torch.Tensor, learned_k: int, uniform_k: int, topk: int) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for learned_row, uniform_row in zip(learned_sorted.cpu(), uniform.cpu()):
        values: list[int] = []
        for index in learned_row[:learned_k].tolist():
            if int(index) not in values:
                values.append(int(index))
        for index in uniform_row[:uniform_k].tolist():
            if int(index) not in values:
                values.append(int(index))
        for index in learned_row.tolist():
            if len(values) >= topk:
                break
            if int(index) not in values:
                values.append(int(index))
        rows.append(torch.tensor(values[:topk], dtype=torch.long))
    return torch.stack(rows, dim=0)


def select_context_tokens(
    *,
    context_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    policy: str,
    topk: int,
    selector: UnifiedPredictiveImportanceSelector | None = None,
    batch: dict[str, Any] | None = None,
    seed: int = 0,
    learned_k: int | None = None,
    uniform_k: int | None = None,
) -> dict[str, Any]:
    if topk <= 0 or topk > context_tokens.shape[1]:
        raise ValueError(f"topk must be in [1, {context_tokens.shape[1]}], got {topk}")
    device = context_tokens.device
    scores: torch.Tensor | None
    if policy == "current_only":
        empty = context_tokens[:, :0]
        return {"selected_tokens": empty, "selected_indices": torch.empty(context_tokens.shape[0], 0, dtype=torch.long, device=device), "scores": None, "selected_scores": None, "policy_name": policy}
    if policy == "random_context_topK":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        indices = torch.stack([torch.randperm(context_tokens.shape[1], generator=generator)[:topk] for _ in range(context_tokens.shape[0])], dim=0).to(device)
        scores = None
    elif policy == "uniform_context_topK":
        indices = uniform_context_indices(context_tokens.shape[0], context_tokens.shape[1], topk, device)
        scores = None
    elif policy == "teacher_context_importance_topK":
        if batch is None or "importance_scores_norm" not in batch:
            raise KeyError("teacher_context_importance_topK requires importance_scores_norm")
        scores = batch["importance_scores_norm"].to(device).float()
        indices = torch.topk(scores, k=topk, dim=1).indices
    elif policy in {"learned_context_selector_topK", "hybrid_context_learned_uniform"}:
        if selector is None:
            raise ValueError(f"{policy} requires a UnifiedPredictiveImportanceSelector")
        selector.eval()
        with torch.no_grad():
            scores = torch.sigmoid(selector(context_tokens=context_tokens, current_tokens=current_tokens, mode="context"))
        if policy == "learned_context_selector_topK":
            indices = torch.topk(scores, k=topk, dim=1).indices
        else:
            lk = int(learned_k if learned_k is not None else topk // 2)
            uk = int(uniform_k if uniform_k is not None else topk - lk)
            learned_sorted = torch.topk(scores, k=min(context_tokens.shape[1], topk * 3), dim=1).indices
            uniform = uniform_context_indices(context_tokens.shape[0], context_tokens.shape[1], uk, device)
            indices = hybrid_context_indices(learned_sorted, uniform, lk, uk, topk).to(device)
    else:
        raise ValueError(f"Unsupported context policy: {policy!r}")
    selected_tokens = gather_tokens_by_indices(context_tokens, indices)
    selected_scores = torch.gather(scores, dim=1, index=indices) if scores is not None else None
    return {
        "selected_tokens": selected_tokens,
        "selected_indices": indices,
        "scores": scores,
        "selected_scores": selected_scores,
        "policy_name": policy,
    }


def context_selection_metrics(selected_indices: torch.Tensor, importance_scores: torch.Tensor | None, num_tokens: int, seed: int = 0) -> dict[str, float | None]:
    if importance_scores is None or selected_indices.shape[1] == 0:
        return {
            "selector_target_top1_overlap": None,
            "selector_target_topk_overlap": None,
            "selected_context_importance_mean": None,
            "random_context_importance_mean": None,
            "selected_vs_random_context_importance_gap": None,
        }
    metrics = compute_teacher_importance_selection_metrics(
        selected_indices,
        importance_scores,
        k=int(selected_indices.shape[1]),
        num_tokens=num_tokens,
        random_seed=seed,
    )
    return {
        "selector_target_top1_overlap": metrics["selector_target_top1_overlap"],
        "selector_target_topk_overlap": metrics["selector_target_topk_overlap"],
        "selected_context_importance_mean": metrics["selected_teacher_importance_mean"],
        "random_context_importance_mean": metrics["random_teacher_importance_mean"],
        "selected_vs_random_context_importance_gap": metrics["selected_vs_random_importance_gap"],
        "context_retention_ratio": metrics["token_retention_ratio"],
    }
