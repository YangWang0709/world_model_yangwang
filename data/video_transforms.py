"""Placeholder video transforms for future dataset work."""

from __future__ import annotations

import torch


def identity_video_transform(video: torch.Tensor) -> torch.Tensor:
    """Return the input unchanged."""

    return video

