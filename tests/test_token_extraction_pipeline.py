"""Tests for the dummy token extraction path."""

from pathlib import Path

import torch

from data.token_shards import save_token_shard, validate_token_shard
from data.toy_data import generate_toy_video_dataset
from data.video_clip_dataset import VideoClipDataset
from encoders.dummy_video_encoder import DummyVideoEncoder


def test_dummy_token_extraction_pipeline(tmp_path: Path) -> None:
    dataset_root = tmp_path / "toy_videos"
    generate_toy_video_dataset(dataset_root, num_samples=2, total_frames=6, image_size=64, seed=7)
    dataset = VideoClipDataset(dataset_root, past_len=4, future_len=2)
    encoder = DummyVideoEncoder(num_tokens=196, token_dim=768)

    batch = [dataset[0], dataset[1]]
    past_video = torch.stack([item["past_video"] for item in batch], dim=0)
    future_video = torch.stack([item["future_video"] for item in batch], dim=0)
    shard = {
        "schema_version": "0.1.0",
        "encoder_name": "dummy_video_encoder",
        "encoder_config": {
            "num_tokens": 196,
            "token_dim": 768,
            "patch_grid_h": 14,
            "patch_grid_w": 14,
        },
        "created_at": "2026-06-10T00:00:00+00:00",
        "split": "toy",
        "sample_ids": [item["sample_id"] for item in batch],
        "task_texts": [item["task_text"] for item in batch],
        "past_tokens": encoder(past_video),
        "future_tokens": encoder(future_video),
        "metadata": [item["metadata"] for item in batch],
    }

    assert validate_token_shard(shard)
    output_path = save_token_shard(tmp_path / "tokens_shard_000000.pt", shard)
    assert output_path.exists()

