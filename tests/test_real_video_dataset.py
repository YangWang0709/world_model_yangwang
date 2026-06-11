"""Tests for RealVideoClipDataset."""

from __future__ import annotations

from pathlib import Path

import torch

from data.real_video_dataset import RealVideoClipDataset
from data.real_video_index import build_real_video_index


def test_real_video_dataset_reads_pt_and_resizes(tmp_path: Path) -> None:
    root = tmp_path / "real_video_minimal"
    raw_dir = root / "raw" / "pt_clips"
    raw_dir.mkdir(parents=True)
    torch.save(
        {
            "video": torch.rand(8, 3, 24, 32),
            "task_text": "predict future visual dynamics",
            "sample_id": "real_000000",
            "fps": 5,
            "source": "real_video_minimal",
        },
        raw_dir / "clip.pt",
    )
    metadata_path = root / "metadata.jsonl"
    build_real_video_index(root / "raw", metadata_path, min_frames=8)

    dataset = RealVideoClipDataset(
        root=root,
        metadata_file="metadata.jsonl",
        past_len=4,
        future_len=4,
        image_size=64,
        split="real_minimal",
    )
    sample = dataset[0]

    assert len(dataset) == 1
    assert list(sample["past_video"].shape) == [4, 3, 64, 64]
    assert list(sample["future_video"].shape) == [4, 3, 64, 64]
    assert sample["past_video"].dtype == torch.float32
    assert sample["future_video"].dtype == torch.float32
    assert float(sample["past_video"].min()) >= 0.0
    assert float(sample["future_video"].max()) <= 1.0
    assert sample["metadata"]["source_type"] == "pt_clip"
