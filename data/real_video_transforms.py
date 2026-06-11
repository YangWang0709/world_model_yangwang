"""Tensor transforms for the Step 9A real-video minimal pipeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def ensure_video_tensor(video: Any) -> torch.Tensor:
    """Return a float32 video tensor with shape [T, C, H, W] and values in [0, 1]."""

    if not isinstance(video, torch.Tensor):
        video = torch.as_tensor(video)
    if video.ndim != 4:
        raise ValueError(f"video must be a 4D tensor, got shape {tuple(video.shape)}")

    if video.shape[1] in (1, 3):
        tensor = video
    elif video.shape[-1] in (1, 3):
        tensor = video.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(
            "video must have channel dimension 1 or 3 in [T, C, H, W] or [T, H, W, C], "
            f"got shape {tuple(video.shape)}"
        )

    if tensor.shape[1] == 1:
        tensor = tensor.expand(-1, 3, -1, -1).contiguous()
    if tensor.shape[1] != 3:
        raise ValueError(f"video channel dimension must be 3, got {tensor.shape[1]}")

    tensor = tensor.detach().cpu().contiguous()
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(torch.float32) / 255.0
    else:
        tensor = tensor.to(torch.float32)
        finite_values = tensor[torch.isfinite(tensor)]
        if finite_values.numel() and float(finite_values.max()) > 1.5:
            tensor = tensor / 255.0

    return tensor.clamp(0.0, 1.0).contiguous()


def validate_video_tensor(video: torch.Tensor, min_frames: int | None = None) -> torch.Tensor:
    """Validate a [T, C, H, W] video tensor and return it unchanged."""

    if not isinstance(video, torch.Tensor):
        raise TypeError("video must be a torch.Tensor")
    if video.ndim != 4:
        raise ValueError(f"video must have shape [T, C, H, W], got {tuple(video.shape)}")
    if video.shape[1] != 3:
        raise ValueError(f"video must have 3 channels, got {video.shape[1]}")
    if min_frames is not None and video.shape[0] < min_frames:
        raise ValueError(f"video has {video.shape[0]} frames but needs at least {min_frames}")
    if not torch.is_floating_point(video):
        raise TypeError("video must be a floating point tensor")
    if not torch.isfinite(video).all():
        raise ValueError("video contains non-finite values")
    min_value = float(video.min())
    max_value = float(video.max())
    if min_value < -1e-6 or max_value > 1.0 + 1e-6:
        raise ValueError(f"video values must be in [0, 1], got min={min_value} max={max_value}")
    return video


def resize_video_tensor(video: torch.Tensor, image_size: int | tuple[int, int]) -> torch.Tensor:
    """Resize [T, C, H, W] video frames with torch interpolate."""

    validate_video_tensor(video)
    if isinstance(image_size, int):
        size = (image_size, image_size)
    else:
        size = tuple(image_size)
        if len(size) != 2:
            raise ValueError(f"image_size must be an int or (height, width), got {image_size}")

    if tuple(video.shape[-2:]) == size:
        return video.contiguous()
    return F.interpolate(video, size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0).contiguous()


def split_past_future(video: torch.Tensor, past_len: int, future_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a video into contiguous past and future clips."""

    if past_len <= 0 or future_len <= 0:
        raise ValueError("past_len and future_len must be positive")
    required_frames = past_len + future_len
    validate_video_tensor(video, min_frames=required_frames)
    past_video = video[:past_len].contiguous()
    future_video = video[past_len:required_frames].contiguous()
    return past_video, future_video
