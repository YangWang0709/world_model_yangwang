"""Loss variants for Step37 factorized selector diagnosis."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


LOSS_VARIANTS = (
    "mse_only",
    "mse_plus_rank",
    "mse_plus_topk_soft",
    "mse_plus_rank_plus_topk_soft",
    "topk_bce_plus_rank",
)


def factorized_selector_loss(
    *,
    output: dict[str, torch.Tensor],
    proxy_patch_target: torch.Tensor,
    proxy_temporal_target: torch.Tensor | None,
    variant: str,
    config: dict[str, Any],
    proxy_temporal_aux_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    if variant not in LOSS_VARIANTS:
        raise ValueError(f"unknown loss variant: {variant!r}")
    logits = output["scores"]
    target = proxy_patch_target
    if logits.shape != target.shape or logits.ndim != 3:
        raise ValueError(f"logits and target must both be [B, T, S], got {tuple(logits.shape)} and {tuple(target.shape)}")
    pred = torch.sigmoid(logits)
    mse = F.mse_loss(pred, target, reduction="mean")
    rank = sampled_pairwise_rank_loss(
        logits,
        target,
        num_pairs=int(config.get("rank_pairs_per_batch", 2048)),
    )
    topk = topk_soft_bce_loss(logits, target, [int(k) for k in config.get("topk_values", [64, 128, 256, 512])])
    if variant == "mse_only":
        total = mse
    elif variant == "mse_plus_rank":
        total = mse + 0.1 * rank
    elif variant == "mse_plus_topk_soft":
        total = mse + 0.1 * topk
    elif variant == "mse_plus_rank_plus_topk_soft":
        total = mse + 0.1 * rank + 0.1 * topk
    else:
        total = topk + 0.1 * rank
    temporal_aux = logits.new_tensor(0.0)
    if float(proxy_temporal_aux_weight) > 0.0:
        if proxy_temporal_target is None:
            raise ValueError("proxy_temporal_target is required for temporal auxiliary loss")
        temporal_logits = output["temporal_scores"]
        if temporal_logits.shape != proxy_temporal_target.shape:
            raise ValueError(
                "temporal_scores and proxy_temporal_target must match, got "
                f"{tuple(temporal_logits.shape)} and {tuple(proxy_temporal_target.shape)}"
            )
        temporal_aux = F.mse_loss(torch.sigmoid(temporal_logits), proxy_temporal_target, reduction="mean")
        total = total + float(proxy_temporal_aux_weight) * temporal_aux
    return {
        "loss": total,
        "mse_loss": mse.detach(),
        "rank_loss": rank.detach(),
        "topk_soft_loss": topk.detach(),
        "temporal_aux_loss": temporal_aux.detach(),
    }


def sampled_pairwise_rank_loss(logits: torch.Tensor, target: torch.Tensor, *, num_pairs: int) -> torch.Tensor:
    if logits.shape != target.shape or logits.ndim != 3:
        raise ValueError("logits and target must both be [B, T, S]")
    batch = int(logits.shape[0])
    flat_logits = logits.reshape(batch, -1)
    flat_target = target.reshape(batch, -1)
    num_tokens = int(flat_logits.shape[1])
    pairs = max(1, int(num_pairs))
    left = torch.randint(0, num_tokens, (batch, pairs), device=logits.device)
    right = torch.randint(0, num_tokens, (batch, pairs), device=logits.device)
    left_score = flat_logits.gather(1, left)
    right_score = flat_logits.gather(1, right)
    left_target = flat_target.gather(1, left)
    right_target = flat_target.gather(1, right)
    target_diff = left_target - right_target
    mask = torch.abs(target_diff) > 1.0e-6
    if not bool(mask.any()):
        return logits.new_tensor(0.0)
    sign = torch.sign(target_diff)
    score_diff = left_score - right_score
    return F.softplus(-score_diff[mask] * sign[mask]).mean()


def topk_soft_bce_loss(logits: torch.Tensor, target: torch.Tensor, topk_values: list[int]) -> torch.Tensor:
    if logits.shape != target.shape or logits.ndim != 3:
        raise ValueError("logits and target must both be [B, T, S]")
    flat_logits = logits.reshape(logits.shape[0], -1)
    flat_target = target.reshape(target.shape[0], -1)
    losses = []
    for k in topk_values:
        kk = max(1, min(int(k), int(flat_target.shape[1])))
        mask = torch.zeros_like(flat_target)
        indices = torch.topk(flat_target, k=kk, dim=1).indices
        mask.scatter_(1, indices, 1.0)
        losses.append(F.binary_cross_entropy_with_logits(flat_logits, mask, reduction="mean"))
    return torch.stack(losses).mean() if losses else logits.new_tensor(0.0)
