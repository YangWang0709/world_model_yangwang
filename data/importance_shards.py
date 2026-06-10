"""Predictive token importance shard helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


IMPORTANCE_SHARD_SCHEMA_VERSION = "0.1.0"

REQUIRED_IMPORTANCE_SHARD_KEYS = (
    "schema_version",
    "importance_method",
    "teacher_checkpoint",
    "teacher_config",
    "source_token_shard",
    "created_at",
    "split",
    "sample_ids",
    "task_texts",
    "importance_scores",
    "importance_scores_norm",
    "base_losses",
    "masked_losses",
    "metadata",
    "mask_config",
)


def utc_now_iso() -> str:
    """Return a UTC ISO timestamp for shard metadata."""

    return datetime.now(timezone.utc).isoformat()


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def validate_importance_shard(shard_dict: dict[str, Any], strict: bool = True) -> bool:
    """Validate a Step 5 predictive importance shard.

    Raises a clear exception on invalid shards and returns True otherwise.
    """

    if not isinstance(shard_dict, dict):
        raise TypeError("importance shard must be a dict")

    missing = [key for key in REQUIRED_IMPORTANCE_SHARD_KEYS if key not in shard_dict]
    if missing:
        raise KeyError(f"importance shard missing required keys: {missing}")

    schema_version = shard_dict["schema_version"]
    if schema_version != IMPORTANCE_SHARD_SCHEMA_VERSION:
        raise ValueError(
            "unsupported schema_version "
            f"{schema_version!r}; expected {IMPORTANCE_SHARD_SCHEMA_VERSION!r}"
        )

    importance_scores = shard_dict["importance_scores"]
    importance_scores_norm = shard_dict["importance_scores_norm"]
    base_losses = shard_dict["base_losses"]
    masked_losses = shard_dict["masked_losses"]
    tensor_fields = {
        "importance_scores": importance_scores,
        "importance_scores_norm": importance_scores_norm,
        "base_losses": base_losses,
        "masked_losses": masked_losses,
    }
    for key, value in tensor_fields.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor")
        if not torch.isfinite(value).all():
            raise ValueError(f"{key} must contain only finite values")

    if importance_scores.ndim != 2:
        raise ValueError(
            f"importance_scores must have shape [B, N], got {tuple(importance_scores.shape)}"
        )
    if importance_scores_norm.shape != importance_scores.shape:
        raise ValueError(
            "importance_scores_norm shape must match importance_scores, "
            f"got {tuple(importance_scores_norm.shape)} and {tuple(importance_scores.shape)}"
        )
    if base_losses.ndim != 1:
        raise ValueError(f"base_losses must have shape [B], got {tuple(base_losses.shape)}")
    if masked_losses.shape != importance_scores.shape:
        raise ValueError(
            "masked_losses must have shape [B, N] matching importance_scores, "
            f"got {tuple(masked_losses.shape)} and {tuple(importance_scores.shape)}"
        )

    batch_size = int(importance_scores.shape[0])
    if int(base_losses.shape[0]) != batch_size:
        raise ValueError("base_losses batch size must match importance_scores")

    for list_key in ("sample_ids", "task_texts", "metadata"):
        value = shard_dict[list_key]
        if not isinstance(value, list):
            raise TypeError(f"{list_key} must be a list")
        if len(value) != batch_size:
            raise ValueError(
                f"{list_key} length {len(value)} must match batch size {batch_size}"
            )

    if strict:
        for string_key in (
            "importance_method",
            "teacher_checkpoint",
            "source_token_shard",
            "created_at",
            "split",
        ):
            if not isinstance(shard_dict[string_key], str):
                raise TypeError(f"{string_key} must be a string")
        if not isinstance(shard_dict["teacher_config"], dict):
            raise TypeError("teacher_config must be a dict")
        if not isinstance(shard_dict["mask_config"], dict):
            raise TypeError("mask_config must be a dict")

    return True


def save_importance_shard(path: str | Path, shard_dict: dict[str, Any]) -> Path:
    """Validate and save an importance shard."""

    validate_importance_shard(shard_dict, strict=True)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shard_dict, output_path)
    return output_path


def load_importance_shard(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate an importance shard."""

    shard = torch.load(Path(path), map_location=map_location)
    validate_importance_shard(shard, strict=True)
    return shard


def summarize_importance_shard(shard_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-serializable importance shard summary."""

    validate_importance_shard(shard_dict, strict=True)
    importance_stats = _tensor_stats(shard_dict["importance_scores"])
    norm_stats = _tensor_stats(shard_dict["importance_scores_norm"])
    return {
        "schema_version": shard_dict["schema_version"],
        "importance_method": shard_dict["importance_method"],
        "num_samples": len(shard_dict["sample_ids"]),
        "num_tokens": int(shard_dict["importance_scores"].shape[1]),
        "importance_scores_shape": list(shard_dict["importance_scores"].shape),
        "importance_scores_norm_shape": list(shard_dict["importance_scores_norm"].shape),
        "base_losses_shape": list(shard_dict["base_losses"].shape),
        "masked_losses_shape": list(shard_dict["masked_losses"].shape),
        "importance_mean": importance_stats["mean"],
        "importance_std": importance_stats["std"],
        "importance_min": importance_stats["min"],
        "importance_max": importance_stats["max"],
        "importance_norm_mean": norm_stats["mean"],
        "importance_norm_std": norm_stats["std"],
        "importance_norm_min": norm_stats["min"],
        "importance_norm_max": norm_stats["max"],
        "base_loss_mean": _tensor_stats(shard_dict["base_losses"])["mean"],
        "masked_loss_mean": _tensor_stats(shard_dict["masked_losses"])["mean"],
        "split": shard_dict["split"],
        "source_token_shard": shard_dict["source_token_shard"],
    }
