from __future__ import annotations

from pathlib import Path

import torch

from data.structured_token_toy import StructuredTokenToyConfig, write_structured_token_shards
from data.token_shards import load_token_shard, validate_token_shard


def test_structured_token_toy_shard_schema_and_metadata(tmp_path: Path) -> None:
    config = StructuredTokenToyConfig(
        num_samples=4,
        num_tokens=12,
        token_dim=24,
        num_key_tokens=3,
        signal_scale=2.0,
        noise_scale=0.01,
        shard_size=2,
    )
    summary = write_structured_token_shards(tmp_path, config)

    assert summary["num_shards"] == 2
    shard = load_token_shard(tmp_path / "tokens_shard_000000.pt")
    assert validate_token_shard(shard)
    assert shard["past_tokens"].shape == (2, 12, 24)
    assert shard["future_tokens"].shape == (2, 12, 24)
    assert "aux_labels" in shard
    assert shard["aux_labels"]["key_token_mask"].shape == (2, 12)
    assert torch.all(shard["aux_labels"]["key_token_mask"].sum(dim=1) == 3)
    assert shard["metadata"][0]["source"] == "structured_token_toy"
    assert len(shard["metadata"][0]["key_token_indices"]) == 3
    assert len(shard["metadata"][0]["key_token_mask"]) == 12
    assert sum(shard["metadata"][0]["key_token_mask"]) == 3


def test_validate_token_shard_accepts_old_schema_without_aux_labels() -> None:
    shard = {
        "schema_version": "0.1.0",
        "encoder_name": "dummy",
        "encoder_config": {},
        "created_at": "2026-06-10T00:00:00+00:00",
        "split": "toy",
        "sample_ids": ["a"],
        "task_texts": ["task"],
        "past_tokens": torch.zeros(1, 2, 3),
        "future_tokens": torch.zeros(1, 2, 3),
        "metadata": [{}],
    }
    assert validate_token_shard(shard)
