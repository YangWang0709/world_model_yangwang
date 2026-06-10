"""Token shard helpers and schema validation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


TOKEN_SHARD_SCHEMA_VERSION = "0.1.0"

REQUIRED_TOKEN_SHARD_KEYS = (
    "schema_version",
    "encoder_name",
    "encoder_config",
    "created_at",
    "split",
    "sample_ids",
    "task_texts",
    "past_tokens",
    "future_tokens",
    "metadata",
)


def token_shard_path(root: str | Path, sample_id: str, suffix: str = ".pt") -> Path:
    """Build a deterministic token shard path without writing any files."""

    return Path(root) / f"{sample_id}{suffix}"


def utc_now_iso() -> str:
    """Return an ISO timestamp for shard metadata."""

    return datetime.now(timezone.utc).isoformat()


def validate_token_shard(shard_dict: dict[str, Any], strict: bool = True) -> bool:
    """Validate the Step 3 token shard schema.

    Raises a clear exception on invalid shards and returns True otherwise.
    """

    if not isinstance(shard_dict, dict):
        raise TypeError("token shard must be a dict")

    missing = [key for key in REQUIRED_TOKEN_SHARD_KEYS if key not in shard_dict]
    if missing:
        raise KeyError(f"token shard missing required keys: {missing}")

    schema_version = shard_dict["schema_version"]
    if schema_version != TOKEN_SHARD_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {schema_version!r}; expected {TOKEN_SHARD_SCHEMA_VERSION!r}"
        )

    past_tokens = shard_dict["past_tokens"]
    future_tokens = shard_dict["future_tokens"]
    if not isinstance(past_tokens, torch.Tensor):
        raise TypeError("past_tokens must be a torch.Tensor")
    if not isinstance(future_tokens, torch.Tensor):
        raise TypeError("future_tokens must be a torch.Tensor")
    if past_tokens.ndim != 3:
        raise ValueError(f"past_tokens must have shape [B, N, D], got {tuple(past_tokens.shape)}")
    if future_tokens.ndim not in (2, 3):
        raise ValueError(
            f"future_tokens must have shape [B, N, D] or [B, D], got {tuple(future_tokens.shape)}"
        )

    batch_size = past_tokens.shape[0]
    if future_tokens.shape[0] != batch_size:
        raise ValueError("future_tokens batch size must match past_tokens")

    for list_key in ("sample_ids", "task_texts", "metadata"):
        value = shard_dict[list_key]
        if not isinstance(value, list):
            raise TypeError(f"{list_key} must be a list")
        if len(value) != batch_size:
            raise ValueError(
                f"{list_key} length {len(value)} must match batch size {batch_size}"
            )

    aux_labels = shard_dict.get("aux_labels")
    if aux_labels is not None:
        if not isinstance(aux_labels, dict):
            raise TypeError("aux_labels must be a dict when present")
        key_token_mask = aux_labels.get("key_token_mask")
        if key_token_mask is not None:
            if not isinstance(key_token_mask, torch.Tensor):
                raise TypeError("aux_labels.key_token_mask must be a torch.Tensor")
            expected_shape = past_tokens.shape[:2]
            if key_token_mask.shape != expected_shape:
                raise ValueError(
                    "aux_labels.key_token_mask must have shape [B, N], "
                    f"got {tuple(key_token_mask.shape)} expected {tuple(expected_shape)}"
                )

    if strict:
        if not isinstance(shard_dict["encoder_config"], dict):
            raise TypeError("encoder_config must be a dict")
        if not isinstance(shard_dict["encoder_name"], str):
            raise TypeError("encoder_name must be a string")
        if not isinstance(shard_dict["split"], str):
            raise TypeError("split must be a string")
        if not isinstance(shard_dict["created_at"], str):
            raise TypeError("created_at must be a string")

    return True


def save_token_shard(path: str | Path, shard_dict: dict[str, Any]) -> Path:
    """Validate and save a token shard."""

    validate_token_shard(shard_dict, strict=True)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shard_dict, output_path)
    return output_path


def load_token_shard(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load and validate a token shard."""

    shard = torch.load(Path(path), map_location=map_location)
    validate_token_shard(shard, strict=True)
    return shard


def summarize_token_shard(shard_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-serializable shard summary."""

    validate_token_shard(shard_dict, strict=True)
    return {
        "schema_version": shard_dict["schema_version"],
        "encoder_name": shard_dict["encoder_name"],
        "num_samples": len(shard_dict["sample_ids"]),
        "past_tokens_shape": list(shard_dict["past_tokens"].shape),
        "future_tokens_shape": list(shard_dict["future_tokens"].shape),
        "future_tokens_rank": int(shard_dict["future_tokens"].ndim),
        "split": shard_dict["split"],
    }
