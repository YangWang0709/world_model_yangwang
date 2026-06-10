"""Tests for token shard save/load/validate helpers."""

from pathlib import Path

import pytest
import torch

from data.token_shards import (
    TOKEN_SHARD_SCHEMA_VERSION,
    load_token_shard,
    save_token_shard,
    summarize_token_shard,
    validate_token_shard,
)


def make_shard() -> dict:
    return {
        "schema_version": TOKEN_SHARD_SCHEMA_VERSION,
        "encoder_name": "dummy_video_encoder",
        "encoder_config": {
            "num_tokens": 196,
            "token_dim": 768,
            "patch_grid_h": 14,
            "patch_grid_w": 14,
        },
        "created_at": "2026-06-10T00:00:00+00:00",
        "split": "toy",
        "sample_ids": ["sample_000000", "sample_000001"],
        "task_texts": ["task", "task"],
        "past_tokens": torch.zeros(2, 196, 768),
        "future_tokens": torch.ones(2, 196, 768),
        "metadata": [{"i": 0}, {"i": 1}],
    }


def test_token_shard_save_load_validate_summary(tmp_path: Path) -> None:
    shard = make_shard()
    path = tmp_path / "tokens_shard_000000.pt"
    save_token_shard(path, shard)
    loaded = load_token_shard(path)
    assert validate_token_shard(loaded)
    summary = summarize_token_shard(loaded)
    assert summary["schema_version"] == TOKEN_SHARD_SCHEMA_VERSION
    assert summary["past_tokens_shape"] == [2, 196, 768]
    assert summary["future_tokens_shape"] == [2, 196, 768]
    assert summary["num_samples"] == 2


def test_token_shard_missing_key_raises() -> None:
    shard = make_shard()
    shard.pop("past_tokens")
    with pytest.raises(KeyError):
        validate_token_shard(shard)

