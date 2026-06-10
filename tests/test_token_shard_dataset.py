"""Tests for TokenShardDataset and its collate function."""

from pathlib import Path

import torch

from data.token_shard_dataset import TokenShardDataset, token_shard_collate_fn
from data.token_shards import save_token_shard


def make_shard(path: Path, batch_size: int = 2) -> Path:
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
        "sample_ids": [f"sample_{idx:06d}" for idx in range(batch_size)],
        "task_texts": ["task"] * batch_size,
        "past_tokens": torch.zeros(batch_size, 196, 768),
        "future_tokens": torch.ones(batch_size, 196, 768),
        "metadata": [{"idx": idx} for idx in range(batch_size)],
    }
    return save_token_shard(path, shard)


def test_token_shard_dataset_and_collate(tmp_path: Path) -> None:
    make_shard(tmp_path / "tokens_shard_000000.pt", batch_size=2)
    dataset = TokenShardDataset(tmp_path)
    assert len(dataset) == 2

    item = dataset[0]
    assert list(item["past_tokens"].shape) == [196, 768]
    assert list(item["future_tokens"].shape) == [196, 768]
    assert item["sample_id"] == "sample_000000"

    batch = token_shard_collate_fn([dataset[0], dataset[1]])
    assert list(batch["past_tokens"].shape) == [2, 196, 768]
    assert list(batch["future_tokens"].shape) == [2, 196, 768]
    assert batch["sample_ids"] == ["sample_000000", "sample_000001"]

