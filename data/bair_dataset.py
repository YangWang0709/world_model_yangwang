"""PyTorch loader for exported BAIR Robot Pushing small .pt clips."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .real_video_index import read_metadata_jsonl
from .real_video_transforms import ensure_video_tensor, resize_video_tensor, split_past_future


class BAIRRobotPushingDataset(Dataset):
    """Read a small exported BAIR subset without TensorFlow dependencies."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        metadata_file: str | Path | None = None,
        past_len: int = 4,
        future_len: int = 4,
        image_size: int = 224,
        max_samples: int | None = None,
    ) -> None:
        if past_len <= 0 or future_len <= 0:
            raise ValueError("past_len and future_len must be positive")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        if metadata_file is None:
            self.metadata_path = self.split_dir / "metadata.jsonl"
        else:
            metadata_path = Path(metadata_file)
            self.metadata_path = metadata_path if metadata_path.is_absolute() else self.root / metadata_path
        self.past_len = past_len
        self.future_len = future_len
        self.image_size = image_size

        if not self.root.exists():
            raise FileNotFoundError(f"BAIR subset root does not exist: {self.root}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"BAIR metadata file does not exist: {self.metadata_path}")

        records = read_metadata_jsonl(self.metadata_path)
        records = [record for record in records if str(record.get("split", split)) == split]
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive when provided")
            records = records[:max_samples]
        if not records:
            raise ValueError(f"No BAIR records found for split {split!r} in {self.metadata_path}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_clip_path(self, record: dict[str, Any]) -> Path:
        raw_path = record.get("clip_path", record.get("path"))
        if raw_path is None:
            raise KeyError("BAIR metadata record missing clip_path/path")
        path = Path(str(raw_path))
        return path if path.is_absolute() else self.metadata_path.parent / path

    def _load_payload(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"BAIR clip does not exist: {path}")
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"BAIR clip payload must be a dict: {path}")
        if "video" not in payload:
            raise KeyError(f"BAIR clip payload missing video: {path}")
        return payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        record = dict(self.records[index])
        clip_path = self._resolve_clip_path(record)
        payload = self._load_payload(clip_path)
        video = resize_video_tensor(ensure_video_tensor(payload["video"]), self.image_size)
        past_video, future_video = split_past_future(video, self.past_len, self.future_len)

        required_frames = self.past_len + self.future_len
        actions = payload.get("actions")
        if isinstance(actions, torch.Tensor):
            actions = actions[:required_frames].to(torch.float32).contiguous()
        endeffector_pos = payload.get("endeffector_pos")
        if isinstance(endeffector_pos, torch.Tensor):
            endeffector_pos = endeffector_pos[:required_frames].to(torch.float32).contiguous()

        metadata = dict(payload.get("metadata", {}))
        metadata.update(record)
        metadata["resolved_path"] = str(clip_path)
        metadata["required_frames"] = required_frames
        return {
            "past_video": past_video,
            "future_video": future_video,
            "task_text": str(payload.get("task_text", record.get("task_text", ""))),
            "sample_id": str(payload.get("sample_id", record.get("sample_id", index))),
            "actions": actions,
            "endeffector_pos": endeffector_pos,
            "metadata": metadata,
        }
