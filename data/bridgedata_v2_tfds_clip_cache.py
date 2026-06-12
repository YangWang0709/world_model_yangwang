"""Clip-cache helpers for Step24 BridgeData TFDS token smoke runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

VIDEO_KEYS = ("context_video", "current_video", "future_video")


def load_clip_cache_npz(path: str | Path) -> dict[str, Any]:
    """Load a small Step24 clip cache without importing TensorFlow."""

    cache_path = Path(path)
    with np.load(cache_path, allow_pickle=False) as payload:
        sample: dict[str, Any] = {key: payload[key] for key in payload.files if key != "metadata_json"}
        raw_metadata = payload["metadata_json"].item() if "metadata_json" in payload.files else "{}"
    sample["metadata"] = json.loads(str(raw_metadata))
    sample["path"] = str(cache_path)
    return sample


def validate_clip_cache(
    sample: dict[str, Any],
    *,
    expected_context: int = 16,
    expected_current: int = 4,
    expected_future: int = 4,
) -> bool:
    """Return whether a sample has Step24 context/current/future videos."""

    try:
        _to_tchw(sample["context_video"], expected_t=expected_context)
        _to_tchw(sample["current_video"], expected_t=expected_current)
        _to_tchw(sample["future_video"], expected_t=expected_future)
    except Exception:
        return False
    metadata = sample.get("metadata", {})
    return not any(
        bool(metadata.get(flag))
        for flag in ("action_used_as_input", "language_used_as_input", "goal_used_as_input")
    )


def clip_cache_to_torch_videos(sample: dict[str, Any], *, image_size: int = 224) -> dict[str, Any]:
    """Convert uint8/float numpy clips to float torch tensors shaped [T, 3, H, W]."""

    import torch

    converted: dict[str, Any] = {"metadata": dict(sample.get("metadata", {}))}
    for key in VIDEO_KEYS:
        array = _to_tchw(sample[key])
        if array.shape[-2:] != (image_size, image_size):
            raise ValueError(f"{key} expected image_size={image_size}, got {tuple(array.shape[-2:])}")
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div(255.0)
        else:
            tensor = tensor.float()
        converted[key] = tensor
    return converted


def _to_tchw(array: Any, *, expected_t: int | None = None) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 4:
        raise ValueError(f"clip video must be rank 4, got shape {tuple(value.shape)}")
    if expected_t is not None and int(value.shape[0]) != expected_t:
        raise ValueError(f"expected {expected_t} frames, got {int(value.shape[0])}")
    if int(value.shape[1]) == 3:
        return value
    if int(value.shape[-1]) == 3:
        return np.transpose(value, (0, 3, 1, 2))
    raise ValueError(f"clip video must be [T,3,H,W] or [T,H,W,3], got {tuple(value.shape)}")
