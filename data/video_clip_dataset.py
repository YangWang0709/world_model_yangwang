"""Dataset for metadata-driven `.pt` video clips."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .toy_data import iter_metadata


class VideoClipDataset(Dataset):
    """Read `.pt` clip files and return past/future video splits."""

    def __init__(
        self,
        root: str | Path,
        metadata_file: str | Path = "metadata.jsonl",
        past_len: int = 4,
        future_len: int = 2,
    ) -> None:
        if past_len <= 0 or future_len <= 0:
            raise ValueError("past_len and future_len must be positive")

        self.root = Path(root)
        self.metadata_path = self.root / metadata_file
        self.past_len = past_len
        self.future_len = future_len

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file does not exist: {self.metadata_path}")

        self.records = list(iter_metadata(self.metadata_path))
        if not self.records:
            raise ValueError(f"No metadata records found in {self.metadata_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        record = dict(self.records[index])
        if "clip_path" not in record:
            raise KeyError(f"metadata record missing clip_path: index={index}")

        clip_path = self.root / record["clip_path"]
        if not clip_path.exists():
            raise FileNotFoundError(f"Clip file does not exist: {clip_path}")
        if clip_path.suffix != ".pt":
            raise ValueError(f"Only .pt clip files are supported in Step 3, got: {clip_path}")

        payload = torch.load(clip_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Clip payload must be a dict: {clip_path}")
        if "video" not in payload:
            raise KeyError(f"Clip payload missing video tensor: {clip_path}")

        video = payload["video"]
        if not isinstance(video, torch.Tensor):
            raise TypeError(f"video must be a torch.Tensor in {clip_path}")
        if video.ndim != 4:
            raise ValueError(f"video must have shape [T, C, H, W], got {tuple(video.shape)} in {clip_path}")
        if video.shape[1] != 3:
            raise ValueError(f"video channel dimension must be 3, got {video.shape[1]} in {clip_path}")
        if not torch.is_floating_point(video):
            raise TypeError(f"video must be a floating point tensor in {clip_path}")

        required_frames = self.past_len + self.future_len
        if video.shape[0] < required_frames:
            raise ValueError(
                f"video has {video.shape[0]} frames but needs at least {required_frames}: {clip_path}"
            )

        past_video = video[: self.past_len].contiguous()
        future_video = video[self.past_len : self.past_len + self.future_len].contiguous()
        task_text = str(payload.get("task_text", record.get("task_text", "")))
        sample_id = str(payload.get("sample_id", record.get("sample_id", index)))

        return {
            "past_video": past_video,
            "future_video": future_video,
            "task_text": task_text,
            "sample_id": sample_id,
            "metadata": record,
        }

