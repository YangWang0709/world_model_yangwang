"""Selector loss variants for BAIR Teacher-importance supervision."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


SUPPORTED_SELECTOR_LOSSES = {
    "mse_only",
    "weighted_mse",
    "mse_plus_pairwise_rank",
    "topk_bce",
    "hybrid_weighted_mse_rank_bce",
}


def _zero_like(logits: torch.Tensor) -> torch.Tensor:
    return logits.sum() * 0.0


def _loss_name(loss_config: dict[str, Any]) -> str:
    name = str(loss_config.get("type", loss_config.get("name", "mse_only")))
    if name not in SUPPORTED_SELECTOR_LOSSES:
        raise ValueError(f"Unsupported selector loss: {name!r}")
    return name


def _topk_labels(target: torch.Tensor, topk: int) -> torch.Tensor:
    if topk <= 0 or topk > target.shape[1]:
        raise ValueError(f"topk must be in [1, {target.shape[1]}], got {topk}")
    labels = torch.zeros_like(target)
    indices = torch.topk(target, k=topk, dim=1).indices
    labels.scatter_(dim=1, index=indices, value=1.0)
    return labels


def _pairwise_rank_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    topk: int,
    num_pairs: int,
    temperature: float,
) -> torch.Tensor:
    if topk <= 0 or topk > target.shape[1]:
        raise ValueError(f"topk must be in [1, {target.shape[1]}], got {topk}")
    if num_pairs <= 0:
        return _zero_like(logits)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    losses = []
    all_indices = torch.arange(target.shape[1], device=target.device)
    target_topk = torch.topk(target, k=topk, dim=1).indices
    for sample_logits, sample_topk in zip(logits, target_topk):
        positive_pool = sample_topk.long()
        negative_mask = torch.ones(target.shape[1], dtype=torch.bool, device=target.device)
        negative_mask[positive_pool] = False
        negative_pool = all_indices[negative_mask]
        if positive_pool.numel() == 0 or negative_pool.numel() == 0:
            continue
        pos_pick = positive_pool[
            torch.randint(positive_pool.numel(), (num_pairs,), device=target.device)
        ]
        neg_pick = negative_pool[
            torch.randint(negative_pool.numel(), (num_pairs,), device=target.device)
        ]
        margin = (sample_logits[pos_pick] - sample_logits[neg_pick]) / float(temperature)
        losses.append(F.softplus(-margin).mean())
    if not losses:
        return _zero_like(logits)
    return torch.stack(losses).mean()


def _topk_bce_loss(logits: torch.Tensor, target: torch.Tensor, topk: int) -> torch.Tensor:
    labels = _topk_labels(target, topk=topk)
    num_tokens = int(target.shape[1])
    pos_weight = logits.new_tensor(float(num_tokens - topk) / float(max(topk, 1)))
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def compute_selector_loss(
    logits: torch.Tensor,
    target_importance_norm: torch.Tensor,
    loss_config: dict[str, Any],
) -> dict[str, Any]:
    """Compute one selector objective from raw selector logits.

    The target is normalized Teacher predictive importance from Step 11D. This
    function intentionally avoids online Teacher calls and avoids quadratic
    pairwise matrices.
    """

    if logits.shape != target_importance_norm.shape:
        raise ValueError(
            "logits and target_importance_norm must have the same shape, "
            f"got {tuple(logits.shape)} and {tuple(target_importance_norm.shape)}"
        )
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B, N], got {tuple(logits.shape)}")

    target = torch.nan_to_num(target_importance_norm.float(), nan=0.0, posinf=1.0, neginf=0.0)
    target = target.clamp(0.0, 1.0)
    scores = torch.sigmoid(logits)
    name = _loss_name(loss_config)

    topk = int(loss_config.get("topk", 16))
    mse_loss = F.mse_loss(scores, target)
    weighted_mse_loss = _zero_like(logits)
    rank_loss = _zero_like(logits)
    topk_bce_loss = _zero_like(logits)

    if name in {"weighted_mse", "hybrid_weighted_mse_rank_bce"}:
        alpha = float(loss_config.get("alpha", loss_config.get("weighted_mse_alpha", 2.0)))
        weights = 1.0 + alpha * target
        weighted_mse_loss = (weights * (scores - target).pow(2)).mean()

    if name in {"mse_plus_pairwise_rank", "hybrid_weighted_mse_rank_bce"}:
        rank_loss = _pairwise_rank_loss(
            logits=logits,
            target=target,
            topk=topk,
            num_pairs=int(loss_config.get("num_pairs", 64)),
            temperature=float(loss_config.get("temperature", 1.0)),
        )

    if name in {"topk_bce", "hybrid_weighted_mse_rank_bce"}:
        topk_bce_loss = _topk_bce_loss(logits=logits, target=target, topk=topk)

    if name == "mse_only":
        loss = mse_loss
    elif name == "weighted_mse":
        loss = weighted_mse_loss
    elif name == "mse_plus_pairwise_rank":
        loss = mse_loss + float(loss_config.get("rank_loss_weight", 0.1)) * rank_loss
    elif name == "topk_bce":
        loss = float(loss_config.get("bce_loss_weight", 1.0)) * topk_bce_loss
    elif name == "hybrid_weighted_mse_rank_bce":
        loss = (
            float(loss_config.get("weighted_mse_loss_weight", 1.0)) * weighted_mse_loss
            + float(loss_config.get("rank_loss_weight", 0.1)) * rank_loss
            + float(loss_config.get("bce_loss_weight", 0.2)) * topk_bce_loss
        )
    else:
        raise AssertionError(f"Unhandled selector loss: {name}")

    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite selector loss for {name}: {loss.item()}")
    return {
        "loss": loss,
        "mse_loss": mse_loss,
        "weighted_mse_loss": weighted_mse_loss,
        "rank_loss": rank_loss,
        "topk_bce_loss": topk_bce_loss,
        "loss_name": name,
    }
