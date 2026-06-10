from __future__ import annotations

import pytest
import torch

from data.importance_shards import (
    IMPORTANCE_SHARD_SCHEMA_VERSION,
    save_importance_shard,
    load_importance_shard,
    summarize_importance_shard,
    validate_importance_shard,
)


def make_valid_shard() -> dict:
    importance = torch.tensor([[0.0, 0.5, 1.0], [0.2, -0.1, 0.3]], dtype=torch.float32)
    return {
        "schema_version": IMPORTANCE_SHARD_SCHEMA_VERSION,
        "importance_method": "teacher_token_occlusion",
        "teacher_checkpoint": "runs/teacher/checkpoint.pt",
        "teacher_config": {
            "token_dim": 4,
            "hidden_dim": 8,
            "output_dim": 4,
            "num_layers": 2,
            "dropout": 0.0,
            "pool": "mean",
        },
        "source_token_shard": "data/token_shards/tokens_shard_000000.pt",
        "created_at": "2026-06-10T00:00:00+00:00",
        "split": "toy",
        "sample_ids": ["a", "b"],
        "task_texts": ["task a", "task b"],
        "importance_scores": importance,
        "importance_scores_norm": torch.tensor([[0.0, 0.5, 1.0], [0.75, 0.0, 1.0]], dtype=torch.float32),
        "base_losses": torch.tensor([0.1, 0.2], dtype=torch.float32),
        "masked_losses": importance + torch.tensor([[0.1], [0.2]], dtype=torch.float32),
        "metadata": [{"index": 0}, {"index": 1}],
        "mask_config": {
            "mask_mode": "zero",
            "mask_value": 0.0,
            "clamp_negative_importance": False,
            "normalize": "minmax_per_sample",
        },
    }


def test_save_load_validate_importance_shard(tmp_path):
    shard = make_valid_shard()
    path = save_importance_shard(tmp_path / "importance_shard_000000.pt", shard)
    loaded = load_importance_shard(path)

    assert validate_importance_shard(loaded)
    summary = summarize_importance_shard(loaded)
    assert summary["importance_scores_shape"] == [2, 3]
    assert summary["base_losses_shape"] == [2]
    assert summary["masked_losses_shape"] == [2, 3]
    assert summary["num_samples"] == 2
    assert summary["num_tokens"] == 3


def test_validate_importance_shard_missing_key_raises():
    shard = make_valid_shard()
    shard.pop("importance_scores")
    with pytest.raises(KeyError):
        validate_importance_shard(shard)
