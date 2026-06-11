"""Tests for Step 9A real-video metadata indexing."""

from __future__ import annotations

from pathlib import Path

import torch

from data.real_video_index import build_real_video_index, read_metadata_jsonl


def _write_pt_clip(path: Path, sample_id: str, frames: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video": torch.rand(frames, 3, 32, 32),
            "task_text": "predict future visual dynamics",
            "sample_id": sample_id,
            "fps": 5,
            "source": "real_video_minimal",
        },
        path,
    )


def test_build_real_video_index_from_pt_clips(tmp_path: Path) -> None:
    raw_dir = tmp_path / "real_video_minimal" / "raw" / "pt_clips"
    _write_pt_clip(raw_dir / "clip_a.pt", "real_000000")
    _write_pt_clip(raw_dir / "clip_b.pt", "real_000001")
    metadata_path = tmp_path / "real_video_minimal" / "metadata.jsonl"

    summary = build_real_video_index(
        input_dir=tmp_path / "real_video_minimal" / "raw",
        output_metadata=metadata_path,
        max_samples=100,
        min_frames=8,
    )
    records = read_metadata_jsonl(metadata_path)

    assert metadata_path.exists()
    assert summary["num_samples"] == 2
    assert summary["source_counts"]["pt_clip"] == 2
    assert len(records) == 2
    assert records[0]["sample_id"] == "real_000000"
    assert records[0]["source_type"] == "pt_clip"
    assert records[0]["path"].startswith("raw/pt_clips/")
    assert records[0]["num_frames"] == 8
    assert records[0]["height"] == 32
    assert records[0]["width"] == 32
