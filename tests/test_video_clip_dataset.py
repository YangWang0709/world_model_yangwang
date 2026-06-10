"""Tests for metadata-driven video clip loading."""

import json
from pathlib import Path

import torch

from data.video_clip_dataset import VideoClipDataset


def write_toy_clip(root: Path, index: int) -> dict:
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    sample_id = f"sample_{index:06d}"
    clip_path = clips_dir / f"{sample_id}.pt"
    payload = {
        "video": torch.full((6, 3, 64, 64), float(index) / 10.0),
        "task_text": "predict what changes in the toy video",
        "sample_id": sample_id,
        "fps": 5,
        "source": "toy_generated",
    }
    torch.save(payload, clip_path)
    return {
        "sample_id": sample_id,
        "clip_path": f"clips/{sample_id}.pt",
        "task_text": payload["task_text"],
        "num_frames": 6,
        "fps": 5,
        "source": "toy_generated",
    }


def test_video_clip_dataset_reads_pt_clips(tmp_path: Path) -> None:
    records = [write_toy_clip(tmp_path, index) for index in range(2)]
    with (tmp_path / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    dataset = VideoClipDataset(tmp_path, past_len=4, future_len=2)
    assert len(dataset) == 2

    sample = dataset[1]
    assert list(sample["past_video"].shape) == [4, 3, 64, 64]
    assert list(sample["future_video"].shape) == [2, 3, 64, 64]
    assert sample["sample_id"] == "sample_000001"
    assert sample["task_text"] == "predict what changes in the toy video"

