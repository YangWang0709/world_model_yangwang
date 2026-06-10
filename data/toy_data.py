"""Tiny generated video data for Step 3 pipeline smoke tests."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import torch


DEFAULT_TASK_TEXT = "predict what changes in the toy video"


def make_moving_square_video(
    sample_index: int,
    total_frames: int = 6,
    image_size: int = 64,
    seed: int = 42,
) -> torch.Tensor:
    """Create a deterministic moving square clip in [0, 1]."""

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if image_size < 16:
        raise ValueError("image_size must be at least 16")

    rng = random.Random(seed + sample_index)
    video = torch.zeros(total_frames, 3, image_size, image_size, dtype=torch.float32)
    square = max(4, image_size // 8)
    max_pos = image_size - square
    start_x = rng.randint(0, max_pos)
    start_y = rng.randint(0, max_pos)
    dx = 1 + sample_index % 5
    dy = 1 + (sample_index * 2) % 5
    color = torch.tensor(
        [
            0.35 + 0.05 * (sample_index % 5),
            0.20 + 0.08 * ((sample_index + 1) % 5),
            0.55 + 0.04 * ((sample_index + 2) % 5),
        ],
        dtype=torch.float32,
    ).clamp(0.0, 1.0)

    for frame_idx in range(total_frames):
        brightness = 0.05 + 0.02 * frame_idx
        video[frame_idx].fill_(brightness)
        x = (start_x + frame_idx * dx) % (max_pos + 1)
        y = (start_y + frame_idx * dy) % (max_pos + 1)
        video[frame_idx, :, y : y + square, x : x + square] = color.view(3, 1, 1)
        stripe = (sample_index + frame_idx) % image_size
        video[frame_idx, :, stripe : stripe + 1, :] += 0.1

    return video.clamp_(0.0, 1.0)


def generate_toy_video_dataset(
    output_dir: str | Path,
    num_samples: int = 16,
    total_frames: int = 6,
    image_size: int = 64,
    seed: int = 42,
    fps: int = 5,
) -> list[dict]:
    """Generate tiny `.pt` clips and a metadata.jsonl manifest."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    root = Path(output_dir)
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for index in range(num_samples):
        sample_id = f"sample_{index:06d}"
        clip_rel = Path("clips") / f"{sample_id}.pt"
        video = make_moving_square_video(
            sample_index=index,
            total_frames=total_frames,
            image_size=image_size,
            seed=seed,
        )
        payload = {
            "video": video,
            "task_text": DEFAULT_TASK_TEXT,
            "sample_id": sample_id,
            "fps": fps,
            "source": "toy_generated",
        }
        torch.save(payload, root / clip_rel)
        records.append(
            {
                "sample_id": sample_id,
                "clip_path": str(clip_rel).replace("\\", "/"),
                "task_text": DEFAULT_TASK_TEXT,
                "num_frames": total_frames,
                "fps": fps,
                "source": "toy_generated",
            }
        )

    metadata_path = root / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return records


def iter_metadata(metadata_path: str | Path) -> Iterable[dict]:
    """Yield JSONL records with clear line-number errors."""

    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc

