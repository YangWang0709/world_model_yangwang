"""Minimal dataset definitions for Step 2 smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset

from .video_clip_dataset import VideoClipDataset


@dataclass(frozen=True)
class DummyVideoDatasetConfig:
    num_samples: int = 8
    clip_len: int = 4
    channels: int = 3
    image_size: int = 224


class DummyVideoDataset(Dataset):
    """Deterministic dummy video dataset for shape and import checks."""

    def __init__(
        self,
        num_samples: int = 8,
        clip_len: int = 4,
        channels: int = 3,
        image_size: int = 224,
    ) -> None:
        self.config = DummyVideoDatasetConfig(
            num_samples=num_samples,
            clip_len=clip_len,
            channels=channels,
            image_size=image_size,
        )

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> Dict[str, object]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        shape = (
            self.config.clip_len,
            self.config.channels,
            self.config.image_size,
            self.config.image_size,
        )
        base_value = float(index) / max(1, self.config.num_samples)
        past_video = torch.full(shape, base_value, dtype=torch.float32)
        future_video = torch.full(shape, base_value + 0.01, dtype=torch.float32)

        return {
            "past_video": past_video,
            "future_video": future_video,
            "task_text": "navigate to the task-relevant object",
            "sample_id": f"dummy_{index:04d}",
        }


__all__ = ["DummyVideoDataset", "DummyVideoDatasetConfig", "VideoClipDataset"]
