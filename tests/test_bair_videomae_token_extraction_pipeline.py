"""Unit tests for Step 11B BAIR token extraction without real VideoMAE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from data.real_video_index import write_metadata_jsonl
from data.token_shards import load_token_shard, validate_token_shard
from scripts.extract_tokens import extract_tokens_from_config


class _FakeVideoMAEEncoder:
    encoder_name = "videomae"
    availability = {"available": True, "reason": "fake unit-test encoder"}

    def is_available(self) -> bool:
        return True

    def encode(self, video_batch: torch.Tensor) -> torch.Tensor:
        batch_size = int(video_batch.shape[0])
        value = video_batch.mean(dim=(1, 2, 3, 4), keepdim=False).reshape(batch_size, 1, 1)
        return value.expand(batch_size, 3, 5).contiguous()


def _write_bair_like_clip(root: Path, split: str, index: int) -> dict[str, Any]:
    split_dir = root / split
    clips_dir = split_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    sample_id = f"bair_{split}_{index:06d}"
    clip_name = f"{sample_id}.pt"
    payload = {
        "video": torch.rand(8, 3, 16, 16),
        "task_text": "predict robot pushing future visual dynamics",
        "sample_id": sample_id,
        "fps": 5,
        "source": "bair_robot_pushing_small",
        "split": split,
        "camera": "image_main",
        "actions": torch.ones(8, 4),
        "endeffector_pos": torch.ones(8, 3),
        "metadata": {"source": "bair_robot_pushing_small", "split": split, "camera": "image_main"},
    }
    torch.save(payload, clips_dir / clip_name)
    return {
        "sample_id": sample_id,
        "source_type": "pt_clip",
        "clip_path": f"clips/{clip_name}",
        "task_text": payload["task_text"],
        "num_frames": 8,
        "fps": 5,
        "height": 16,
        "width": 16,
        "source": "bair_robot_pushing_small",
        "split": split,
        "camera": "image_main",
        "has_action": True,
        "has_endeffector_pos": True,
    }


def test_bair_videomae_token_extraction_pipeline_uses_bair_splits(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    root = tmp_path / "bair_robot_pushing_small_subset"
    train_records = [_write_bair_like_clip(root, "train", index) for index in range(2)]
    test_records = [_write_bair_like_clip(root, "test", index) for index in range(1)]
    write_metadata_jsonl(root / "train" / "metadata.jsonl", train_records)
    write_metadata_jsonl(root / "test" / "metadata.jsonl", test_records)

    def fake_builder(config: dict[str, Any]) -> _FakeVideoMAEEncoder:
        assert config["name"] == "videomae"
        assert config["allow_download"] is False
        return _FakeVideoMAEEncoder()

    monkeypatch.setattr("scripts.extract_tokens.build_frozen_video_encoder", fake_builder)

    output_root = tmp_path / "token_shards" / "bair_videomae_smoke"
    config = {
        "dataset": {
            "name": "bair_robot_pushing_small_subset",
            "root": str(root),
            "train_metadata_file": str(root / "train" / "metadata.jsonl"),
            "test_metadata_file": str(root / "test" / "metadata.jsonl"),
            "split_names": ["train", "test"],
            "past_len": 4,
            "future_len": 4,
            "image_size": 16,
            "max_train_samples": 2,
            "max_test_samples": 1,
        },
        "encoder": {
            "name": "videomae",
            "model_name_or_path": str(tmp_path / "fake_videomae"),
            "allow_download": False,
            "local_files_only": True,
            "output_mode": "last_hidden_state",
            "num_frames": 4,
        },
        "fallback": {"allow_dummy_fallback": False},
        "extraction": {
            "batch_size": 1,
            "num_workers": 0,
            "device": "cpu",
            "shard_size": 1,
            "seed": 42,
            "output_root": str(output_root),
            "train_output_dir": str(output_root / "train"),
            "test_output_dir": str(output_root / "test"),
        },
    }
    config_path = tmp_path / "token_extraction_bair_videomae_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = extract_tokens_from_config(config_path=config_path, overwrite=True)
    train_shard = load_token_shard(output_root / "train" / "tokens_shard_000000.pt")
    test_shard = load_token_shard(output_root / "test" / "tokens_shard_000000.pt")

    assert summary["total_samples"] == 3
    assert summary["split_summaries"]["train"]["num_shards"] == 2
    assert summary["split_summaries"]["test"]["num_shards"] == 1
    assert validate_token_shard(train_shard)
    assert validate_token_shard(test_shard)
    assert train_shard["split"] == "train"
    assert test_shard["split"] == "test"
    assert train_shard["sample_ids"][0] == "bair_train_000000"
    assert test_shard["sample_ids"][0] == "bair_test_000000"
    assert list(train_shard["past_tokens"].shape) == [1, 3, 5]
    assert train_shard["encoder_name"] == "videomae"
    assert train_shard["encoder_config"]["used_fallback"] is False
    assert train_shard["metadata"][0]["source"] == "bair_robot_pushing_small"
    assert train_shard["metadata"][0]["action_shape"] == [8, 4]
    assert train_shard["metadata"][0]["endeffector_pos_shape"] == [8, 3]
