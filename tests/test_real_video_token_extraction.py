"""Tests for Step 9A real-video dummy token extraction."""

from __future__ import annotations

from pathlib import Path

import yaml

from data.real_video_index import build_real_video_index
from data.token_shards import load_token_shard, validate_token_shard
from scripts.create_real_video_minimal_subset import create_real_video_minimal_subset
from scripts.extract_tokens import extract_tokens_from_config


def test_real_video_dummy_token_extraction(tmp_path: Path) -> None:
    root = tmp_path / "real_video_minimal"
    create_real_video_minimal_subset(root / "raw" / "pt_clips", num_samples=4, total_frames=8, image_size=32, seed=7)
    metadata_path = root / "metadata.jsonl"
    build_real_video_index(root / "raw", metadata_path, max_samples=4, min_frames=8)
    output_dir = tmp_path / "token_shards" / "real_video_dummy"
    config = {
        "dataset": {
            "name": "real_video_minimal",
            "root": str(root),
            "metadata_file": str(metadata_path),
            "split": "real_minimal",
            "past_len": 4,
            "future_len": 4,
            "image_size": 32,
        },
        "encoder": {
            "name": "dummy_video_encoder",
            "num_tokens": 4,
            "token_dim": 8,
            "patch_grid_h": 2,
            "patch_grid_w": 2,
        },
        "extraction": {
            "batch_size": 2,
            "num_workers": 0,
            "device": "cpu",
            "output_dir": str(output_dir),
            "shard_size": 2,
            "seed": 42,
            "max_samples": 4,
        },
    }
    config_path = tmp_path / "real_video_dummy.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = extract_tokens_from_config(config_path=config_path, overwrite=True)
    shard_paths = sorted(output_dir.glob("tokens_shard_*.pt"))
    shard = load_token_shard(shard_paths[0], map_location="cpu")

    assert summary["dataset_size"] == 4
    assert summary["num_shards"] == 2
    assert len(shard_paths) == 2
    assert validate_token_shard(shard)
    assert list(shard["past_tokens"].shape) == [2, 4, 8]
    assert list(shard["future_tokens"].shape) == [2, 4, 8]
    assert shard["split"] == "real_minimal"
    assert shard["metadata"][0]["source_type"] == "pt_clip"
