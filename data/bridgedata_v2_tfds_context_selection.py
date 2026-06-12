"""Context selection policies for Step26 BridgeData TFDS bottleneck smoke."""

from __future__ import annotations

from typing import Any

import torch


CONTEXT_FRAMES = 16
TOKENS_PER_FRAME = 392
TOKEN_DIM = 768
NUM_CONTEXT_TOKENS = CONTEXT_FRAMES * TOKENS_PER_FRAME


def flatten_context_tokens(context_tokens: torch.Tensor) -> torch.Tensor:
    if list(context_tokens.shape) != [CONTEXT_FRAMES, TOKENS_PER_FRAME, TOKEN_DIM]:
        raise ValueError(f"unexpected context_tokens shape: {list(context_tokens.shape)}")
    return context_tokens.detach().to(dtype=torch.float32, device="cpu").reshape(NUM_CONTEXT_TOKENS, TOKEN_DIM)


def flatten_context_importance(context_importance: torch.Tensor) -> torch.Tensor:
    if list(context_importance.shape) != [CONTEXT_FRAMES, TOKENS_PER_FRAME]:
        raise ValueError(f"unexpected context_importance shape: {list(context_importance.shape)}")
    return context_importance.detach().to(dtype=torch.float32, device="cpu").reshape(NUM_CONTEXT_TOKENS)


def select_context_tokens(sample: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    policy_name = str(policy["name"])
    flat_tokens = flatten_context_tokens(sample["context_tokens"])
    flat_importance = flatten_context_importance(sample["context_importance"])
    current_shape = list(sample["current_tokens"].shape)
    if current_shape != [4, 392, 768]:
        raise ValueError(f"current tokens must stay full, got {current_shape}")

    if policy_name == "current_only" or not bool(policy.get("use_context", True)):
        indices = torch.empty((0,), dtype=torch.long)
    elif str(policy.get("selection")) == "random":
        topk = _bounded_topk(policy.get("topk"), NUM_CONTEXT_TOKENS)
        generator = torch.Generator().manual_seed(int(policy.get("seed", 42)))
        indices = torch.randperm(NUM_CONTEXT_TOKENS, generator=generator)[:topk].to(dtype=torch.long)
    elif str(policy.get("selection")) == "importance_topk":
        topk = _bounded_topk(policy.get("topk"), NUM_CONTEXT_TOKENS)
        indices = torch.topk(flat_importance, k=topk, largest=True, sorted=True).indices.to(dtype=torch.long)
    elif str(policy.get("selection")) == "full":
        indices = torch.arange(NUM_CONTEXT_TOKENS, dtype=torch.long)
    else:
        raise ValueError(f"unsupported context policy: {policy!r}")

    selected_tokens = flat_tokens.index_select(0, indices) if indices.numel() else flat_tokens.new_zeros((0, TOKEN_DIM))
    selected_values = (
        flat_importance.index_select(0, indices) if indices.numel() else flat_importance.new_zeros((0,))
    )
    total_mass = float(flat_importance.sum().item())
    selected_mass = float(selected_values.sum().item())
    mass_ratio = selected_mass / total_mass if total_mass > 0 else 0.0
    selection = {
        "policy_name": policy_name,
        "selected_context_tokens": selected_tokens.contiguous(),
        "selected_indices": indices.contiguous(),
        "selected_importance_values": selected_values.contiguous(),
        "selected_importance_mass": mass_ratio,
        "selected_importance_sum": selected_mass,
        "selected_importance_mean": float(selected_values.mean().item()) if selected_values.numel() else 0.0,
        "topk": int(indices.numel()) if policy.get("topk") is not None else None,
        "num_selected": int(indices.numel()),
        "current_tokens_kept_full": True,
        "deployable": bool(policy.get("deployable", True)),
        "temporal_selected_histogram": temporal_selected_histogram(indices),
    }
    return selection


def summarize_selection(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_name": selection["policy_name"],
        "num_selected": int(selection["num_selected"]),
        "topk": selection["topk"],
        "selected_context_shape": list(selection["selected_context_tokens"].shape),
        "selected_importance_mass": float(selection["selected_importance_mass"]),
        "selected_importance_mean": float(selection["selected_importance_mean"]),
        "current_tokens_kept_full": bool(selection["current_tokens_kept_full"]),
        "deployable": bool(selection.get("deployable", True)),
        "temporal_selected_histogram": list(selection["temporal_selected_histogram"]),
    }


def temporal_selected_histogram(indices: torch.Tensor) -> list[int]:
    if indices.numel() == 0:
        return [0 for _ in range(CONTEXT_FRAMES)]
    frames = torch.div(indices.to(dtype=torch.long), TOKENS_PER_FRAME, rounding_mode="floor")
    return torch.bincount(frames, minlength=CONTEXT_FRAMES).to(dtype=torch.long).tolist()


def _bounded_topk(value: Any, maximum: int) -> int:
    topk = int(value)
    if topk < 0:
        raise ValueError("topk must be non-negative")
    return min(topk, maximum)
