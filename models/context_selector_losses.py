"""Context selector loss variants for Step18 oracle-gap diagnostics."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


SUPPORTED_CONTEXT_SELECTOR_LOSSES = {
    "weighted_mse",
    "topk_bce",
    "pairwise_rank",
    "hybrid_weighted_mse_rank_bce",
    "temporal_block_balanced_topk",
}


def _zero_like(logits: torch.Tensor) -> torch.Tensor:
    return logits.sum() * 0.0


def _loss_name(loss_config: dict[str, Any]) -> str:
    name = str(loss_config.get("loss_type", loss_config.get("type", loss_config.get("name", "weighted_mse"))))
    if name not in SUPPORTED_CONTEXT_SELECTOR_LOSSES:
        raise ValueError(f"Unsupported context selector loss: {name!r}")
    return name


def _sanitize_target(target_importance_norm: torch.Tensor) -> torch.Tensor:
    target = torch.nan_to_num(target_importance_norm.float(), nan=0.0, posinf=1.0, neginf=0.0)
    return target.clamp(0.0, 1.0)


def _topk_labels(target: torch.Tensor, topk: int) -> torch.Tensor:
    if target.ndim != 2:
        raise ValueError(f"target must be [B, N], got {tuple(target.shape)}")
    if topk <= 0 or topk > target.shape[1]:
        raise ValueError(f"topk must be in [1, {target.shape[1]}], got {topk}")
    labels = torch.zeros_like(target)
    labels.scatter_(dim=1, index=torch.topk(target, k=topk, dim=1).indices, value=1.0)
    return labels


def _pairwise_rank_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
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

    losses: list[torch.Tensor] = []
    all_indices = torch.arange(target.shape[1], device=target.device)
    target_topk = torch.topk(target, k=topk, dim=1).indices
    for sample_logits, sample_topk in zip(logits, target_topk):
        positive_pool = sample_topk.long()
        negative_mask = torch.ones(target.shape[1], dtype=torch.bool, device=target.device)
        negative_mask[positive_pool] = False
        negative_pool = all_indices[negative_mask]
        if positive_pool.numel() == 0 or negative_pool.numel() == 0:
            continue
        pos_pick = positive_pool[torch.randint(positive_pool.numel(), (num_pairs,), device=target.device)]
        neg_pick = negative_pool[torch.randint(negative_pool.numel(), (num_pairs,), device=target.device)]
        margin = (sample_logits[pos_pick] - sample_logits[neg_pick]) / float(temperature)
        losses.append(F.softplus(-margin).mean())
    if not losses:
        return _zero_like(logits)
    return torch.stack(losses).mean()


def _topk_bce_loss(logits: torch.Tensor, target: torch.Tensor, *, topk: int, labels: torch.Tensor | None = None) -> torch.Tensor:
    labels = _topk_labels(target, topk=topk) if labels is None else labels.float()
    if labels.shape != logits.shape:
        raise ValueError("BCE labels must match logits")
    pos_count = labels.sum().clamp_min(1.0)
    neg_count = (labels.numel() - labels.sum()).clamp_min(1.0)
    pos_weight = logits.new_tensor(float(neg_count.item()) / float(pos_count.item()))
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def _default_temporal_block_ids(num_tokens: int, temporal_blocks: int, device: torch.device) -> torch.Tensor:
    if temporal_blocks <= 0:
        raise ValueError("temporal_blocks must be positive")
    return torch.arange(num_tokens, device=device).mul(int(temporal_blocks)).floor_divide(int(num_tokens)).clamp_max(temporal_blocks - 1)


def _block_balanced_labels(
    target: torch.Tensor,
    temporal_block_ids: torch.Tensor | None,
    *,
    temporal_blocks: int,
    topk_per_block: int,
    total_topk: int,
) -> torch.Tensor:
    if topk_per_block <= 0:
        raise ValueError("topk_per_block must be positive")
    if total_topk <= 0 or total_topk > target.shape[1]:
        raise ValueError(f"total_topk must be in [1, {target.shape[1]}], got {total_topk}")
    block_ids = temporal_block_ids
    if block_ids is None:
        block_ids = _default_temporal_block_ids(target.shape[1], temporal_blocks, target.device)
    block_ids = block_ids.to(target.device).long()
    if block_ids.ndim != 1 or block_ids.numel() != target.shape[1]:
        raise ValueError("temporal_block_ids must be [N_ctx]")

    labels = torch.zeros_like(target)
    for block_id in range(int(temporal_blocks)):
        mask = block_ids == block_id
        count = int(mask.sum().item())
        if count == 0:
            continue
        k = min(int(topk_per_block), count)
        block_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        local = torch.topk(target[:, block_indices], k=k, dim=1).indices
        global_indices = block_indices[local]
        labels.scatter_(dim=1, index=global_indices, value=1.0)

    selected = int(labels[0].sum().item()) if labels.shape[0] else 0
    if selected < total_topk:
        fill = _topk_labels(target, topk=total_topk)
        labels = torch.maximum(labels, fill)
        # If the union exceeds total_topk, keep the highest target-labelled entries.
        if int(labels[0].sum().item()) > total_topk:
            masked = target.masked_fill(labels <= 0, -torch.inf)
            labels = torch.zeros_like(target)
            labels.scatter_(dim=1, index=torch.topk(masked, k=total_topk, dim=1).indices, value=1.0)
    return labels


def _temporal_balance_penalty(
    logits: torch.Tensor,
    temporal_block_ids: torch.Tensor | None,
    *,
    temporal_blocks: int,
) -> torch.Tensor:
    block_ids = temporal_block_ids
    if block_ids is None:
        block_ids = _default_temporal_block_ids(logits.shape[1], temporal_blocks, logits.device)
    block_ids = block_ids.to(logits.device).long()
    probs = torch.sigmoid(logits)
    block_means = []
    for block_id in range(int(temporal_blocks)):
        mask = block_ids == block_id
        if bool(mask.any()):
            block_means.append(probs[:, mask].mean(dim=1))
    if len(block_means) <= 1:
        return _zero_like(logits)
    stacked = torch.stack(block_means, dim=1)
    return stacked.var(dim=1, unbiased=False).mean()


def compute_context_selector_loss(
    logits: torch.Tensor,
    target_importance_norm: torch.Tensor,
    loss_config: dict[str, Any],
    temporal_block_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute a context-only selector objective from raw logits."""

    if logits.shape != target_importance_norm.shape:
        raise ValueError(
            "logits and target_importance_norm must have the same shape, "
            f"got {tuple(logits.shape)} and {tuple(target_importance_norm.shape)}"
        )
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B, N_ctx], got {tuple(logits.shape)}")

    target = _sanitize_target(target_importance_norm)
    scores = torch.sigmoid(logits)
    name = _loss_name(loss_config)
    topk = int(loss_config.get("topk", 32))
    mse_loss = F.mse_loss(scores, target)
    weighted_mse_loss = _zero_like(logits)
    bce_loss = _zero_like(logits)
    rank_loss = _zero_like(logits)
    temporal_balance_loss = _zero_like(logits)

    if name in {"weighted_mse", "hybrid_weighted_mse_rank_bce", "temporal_block_balanced_topk"}:
        alpha = float(loss_config.get("alpha", loss_config.get("weighted_mse_alpha", 2.0)))
        weighted_mse_loss = ((1.0 + alpha * target) * (scores - target).pow(2)).mean()

    if name in {"topk_bce", "hybrid_weighted_mse_rank_bce"}:
        bce_loss = _topk_bce_loss(logits, target, topk=topk)

    if name in {"pairwise_rank", "hybrid_weighted_mse_rank_bce", "temporal_block_balanced_topk"}:
        rank_loss = _pairwise_rank_loss(
            logits,
            target,
            topk=topk,
            num_pairs=int(loss_config.get("num_pairs", 128)),
            temperature=float(loss_config.get("temperature", 1.0)),
        )

    if name == "temporal_block_balanced_topk":
        temporal_blocks = int(loss_config.get("temporal_blocks", 8))
        balanced_labels = _block_balanced_labels(
            target,
            temporal_block_ids,
            temporal_blocks=temporal_blocks,
            topk_per_block=int(loss_config.get("topk_per_block", max(1, topk // max(temporal_blocks, 1)))),
            total_topk=topk,
        )
        bce_loss = _topk_bce_loss(logits, target, topk=topk, labels=balanced_labels)
        temporal_balance_loss = _temporal_balance_penalty(
            logits,
            temporal_block_ids,
            temporal_blocks=temporal_blocks,
        )

    if name == "weighted_mse":
        loss = weighted_mse_loss
    elif name == "topk_bce":
        loss = float(loss_config.get("bce_loss_weight", 1.0)) * bce_loss
    elif name == "pairwise_rank":
        loss = float(loss_config.get("rank_loss_weight", 0.1)) * rank_loss
    elif name == "hybrid_weighted_mse_rank_bce":
        loss = (
            float(loss_config.get("weighted_mse_loss_weight", 1.0)) * weighted_mse_loss
            + float(loss_config.get("rank_loss_weight", 0.1)) * rank_loss
            + float(loss_config.get("bce_loss_weight", 0.2)) * bce_loss
        )
    elif name == "temporal_block_balanced_topk":
        loss = (
            float(loss_config.get("weighted_mse_loss_weight", 1.0)) * weighted_mse_loss
            + float(loss_config.get("rank_loss_weight", 0.1)) * rank_loss
            + float(loss_config.get("bce_loss_weight", 0.2)) * bce_loss
            + float(loss_config.get("temporal_balance_loss_weight", 0.01)) * temporal_balance_loss
        )
    else:
        raise AssertionError(f"Unhandled context selector loss: {name}")

    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite context selector loss for {name}: {loss.item()}")
    return {
        "loss": loss,
        "mse_loss": mse_loss,
        "weighted_mse_loss": weighted_mse_loss,
        "bce_loss": bce_loss,
        "rank_loss": rank_loss,
        "temporal_balance_loss": temporal_balance_loss,
        "loss_name": name,
    }
