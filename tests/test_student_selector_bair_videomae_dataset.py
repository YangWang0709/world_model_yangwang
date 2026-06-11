from __future__ import annotations

from pathlib import Path

import torch

from data.importance_shards import IMPORTANCE_SHARD_SCHEMA_VERSION, save_importance_shard
from data.student_selector_dataset import StudentSelectorDataset, student_selector_collate_fn
from data.token_shards import TOKEN_SHARD_SCHEMA_VERSION, save_token_shard


def _write_fake_bair_pair(root: Path, split: str) -> tuple[Path, Path]:
    token_dir = root / split / "tokens"
    importance_dir = root / split / "importance"
    sample_ids = [f"{split}_sample_b", f"{split}_sample_a"]
    token_shard = {
        "schema_version": TOKEN_SHARD_SCHEMA_VERSION,
        "encoder_name": "videomae",
        "encoder_config": {"num_tokens": 392, "token_dim": 768},
        "created_at": "2026-06-11T00:00:00+00:00",
        "split": split,
        "sample_ids": sample_ids,
        "task_texts": ["push b", "push a"],
        "past_tokens": torch.randn(2, 392, 768),
        "future_tokens": torch.randn(2, 392, 768),
        "metadata": [{"sample_id": sample_ids[0]}, {"sample_id": sample_ids[1]}],
    }
    importance_shard = {
        "schema_version": IMPORTANCE_SHARD_SCHEMA_VERSION,
        "importance_method": "teacher_token_occlusion",
        "teacher_checkpoint": "teacher.pt",
        "teacher_config": {"token_dim": 768},
        "source_token_shard": "tokens.pt",
        "created_at": "2026-06-11T00:00:00+00:00",
        "split": split,
        "sample_ids": [sample_ids[1], sample_ids[0]],
        "task_texts": ["push a", "push b"],
        "importance_scores": torch.randn(2, 392),
        "importance_scores_norm": torch.rand(2, 392),
        "base_losses": torch.zeros(2),
        "masked_losses": torch.ones(2, 392),
        "metadata": [{"sample_id": sample_ids[1]}, {"sample_id": sample_ids[0]}],
        "mask_config": {"mask_mode": "zero"},
    }
    save_token_shard(token_dir / "tokens_shard_000000.pt", token_shard)
    save_importance_shard(importance_dir / "importance_shard_000000.pt", importance_shard)
    return token_dir, importance_dir


def test_bair_student_selector_dataset_aligns_without_key_masks(tmp_path: Path):
    token_dir, importance_dir = _write_fake_bair_pair(tmp_path, "train")

    dataset = StudentSelectorDataset(
        token_dir,
        importance_dir,
        require_key_token_mask=False,
    )
    batch = student_selector_collate_fn([dataset[0], dataset[1]])

    assert len(dataset) == 2
    assert dataset[0]["sample_id"] == "train_sample_a"
    assert dataset[0]["key_token_mask"] is None
    assert dataset[0]["past_tokens"].shape == (392, 768)
    assert dataset[0]["importance_scores_norm"].shape == (392,)
    assert batch["past_tokens"].shape == (2, 392, 768)
    assert batch["importance_scores"].shape == (2, 392)
    assert batch["importance_scores_norm"].shape == (2, 392)
    assert batch["key_token_mask"] is None
    assert batch["has_key_token_mask"] is False


def test_bair_student_selector_dataset_respects_max_samples(tmp_path: Path):
    token_dir, importance_dir = _write_fake_bair_pair(tmp_path, "test")

    dataset = StudentSelectorDataset(
        token_dir,
        importance_dir,
        require_key_token_mask=False,
        max_samples=1,
    )

    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "test_sample_a"
