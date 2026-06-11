"""Metadata-driven real-video clip dataset for Step 9A."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .real_video_index import FRAME_EXTENSIONS, read_metadata_jsonl
from .real_video_transforms import ensure_video_tensor, resize_video_tensor, split_past_future


class RealVideoClipDataset(Dataset):
    """Read .pt clips, frame folders, or optional video files on demand."""

    def __init__(
        self,
        root: str | Path,
        metadata_file: str | Path = "metadata.jsonl",
        past_len: int = 4,
        future_len: int = 4,
        image_size: int = 224,
        split: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        if past_len <= 0 or future_len <= 0:
            raise ValueError("past_len and future_len must be positive")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        self.root = Path(root)
        metadata_path = Path(metadata_file)
        self.metadata_path = metadata_path if metadata_path.is_absolute() else self.root / metadata_path
        self.past_len = past_len
        self.future_len = future_len
        self.image_size = image_size
        self.split = split

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file does not exist: {self.metadata_path}")

        records = read_metadata_jsonl(self.metadata_path)
        if split is not None:
            records = [record for record in records if record.get("split") == split]
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive when provided")
            records = records[:max_samples]
        if not records:
            raise ValueError(f"No metadata records found in {self.metadata_path}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_record_path(self, record: dict[str, Any]) -> Path:
        if "path" not in record:
            raise KeyError("metadata record missing path")
        path = Path(str(record["path"]))
        return path if path.is_absolute() else self.root / path

    def _load_pt_clip(self, path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f".pt clip payload must be a dict: {path}")
        if "video" not in payload:
            raise KeyError(f".pt clip payload missing video tensor: {path}")
        return ensure_video_tensor(payload["video"]), payload

    def _load_frame_folder(self, path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
        frame_paths = sorted(child for child in path.iterdir() if child.suffix.lower() in FRAME_EXTENSIONS)
        if not frame_paths:
            raise ValueError(f"Frame folder has no supported frames: {path}")
        try:
            from PIL import Image
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("Frame folders require PIL/Pillow and numpy") from exc

        frames: list[torch.Tensor] = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                array = np.asarray(image.convert("RGB"))
            frames.append(torch.from_numpy(array.copy()))
        video = torch.stack(frames, dim=0)
        return ensure_video_tensor(video), {"frame_count": len(frame_paths)}

    def _load_video_file(self, path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
        required_frames = self.past_len + self.future_len
        try:
            import imageio.v3 as iio  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("video_file samples require optional imageio support in Step 9A") from exc

        frames: list[torch.Tensor] = []
        try:
            for frame_index, frame in enumerate(iio.imiter(path)):
                if frame_index >= required_frames:
                    break
                frames.append(torch.as_tensor(frame))
        except Exception as exc:  # pragma: no cover - optional backend path
            raise RuntimeError(f"Could not decode optional video file {path}: {exc}") from exc
        if len(frames) < required_frames:
            raise ValueError(f"video file has {len(frames)} decoded frames but needs {required_frames}: {path}")
        return ensure_video_tensor(torch.stack(frames, dim=0)), {"decoded_frames": len(frames)}

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        record = dict(self.records[index])
        source_type = str(record.get("source_type", ""))
        clip_path = self._resolve_record_path(record)
        if not clip_path.exists():
            raise FileNotFoundError(f"Real-video source path does not exist: {clip_path}")

        payload: dict[str, Any] = {}
        if source_type == "pt_clip":
            video, payload = self._load_pt_clip(clip_path)
        elif source_type == "frame_folder":
            video, payload = self._load_frame_folder(clip_path)
        elif source_type == "video_file":
            video, payload = self._load_video_file(clip_path)
        else:
            raise ValueError(f"Unsupported source_type {source_type!r} in record {record.get('sample_id')}")

        required_frames = self.past_len + self.future_len
        video = resize_video_tensor(video, self.image_size)
        past_video, future_video = split_past_future(video, self.past_len, self.future_len)
        task_text = str(payload.get("task_text", record.get("task_text", "")))
        sample_id = str(payload.get("sample_id", record.get("sample_id", index)))

        metadata = dict(record)
        metadata["resolved_path"] = str(clip_path)
        metadata["required_frames"] = required_frames
        return {
            "past_video": past_video,
            "future_video": future_video,
            "task_text": task_text,
            "sample_id": sample_id,
            "metadata": metadata,
        }
