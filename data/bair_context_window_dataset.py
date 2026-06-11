"""Loader for exported BAIR context/current/future window clips."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from data.real_video_index import read_metadata_jsonl


class BAIRContextWindowDataset(Dataset):
    """Read Step17 BAIR 16-frame windows exported as .pt files."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        metadata_file: str | Path | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.metadata_path = Path(metadata_file) if metadata_file is not None else self.split_dir / "metadata.jsonl"
        if not self.metadata_path.is_absolute():
            self.metadata_path = self.root / self.metadata_path
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"BAIR context metadata does not exist: {self.metadata_path}")
        records = [record for record in read_metadata_jsonl(self.metadata_path) if str(record.get("split", split)) == split]
        if max_samples is not None:
            records = records[: int(max_samples)]
        if not records:
            raise ValueError(f"No BAIR context records found for split {split!r}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_clip_path(self, record: dict[str, Any]) -> Path:
        raw = record.get("clip_path", record.get("path"))
        if raw is None:
            raise KeyError("BAIR context metadata missing clip_path/path")
        path = Path(str(raw))
        return path if path.is_absolute() else self.metadata_path.parent / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        clip_path = self._resolve_clip_path(record)
        payload = torch.load(clip_path, map_location="cpu")
        for key in ("context_video", "current_video", "future_video"):
            if key not in payload:
                raise KeyError(f"BAIR context clip missing {key}: {clip_path}")
        metadata = dict(payload.get("metadata", {}))
        metadata.update(record)
        metadata["resolved_path"] = str(clip_path)
        return {
            "context_video": payload["context_video"].float().contiguous(),
            "current_video": payload["current_video"].float().contiguous(),
            "future_video": payload["future_video"].float().contiguous(),
            "actions": payload.get("actions"),
            "endeffector_pos": payload.get("endeffector_pos"),
            "task_text": str(payload.get("task_text", record.get("task_text", ""))),
            "sample_id": str(payload.get("sample_id", record.get("sample_id", index))),
            "metadata": metadata,
        }
