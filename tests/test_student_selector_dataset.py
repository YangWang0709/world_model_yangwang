from __future__ import annotations

from pathlib import Path

import torch

from data.importance_shards import IMPORTANCE_SHARD_SCHEMA_VERSION, save_importance_shard
from data.student_selector_dataset import StudentSelectorDataset, student_selector_collate_fn
from data.token_shards import TOKEN_SHARD_SCHEMA_VERSION, save_token_shard


def test_student_selector_dataset_aligns_by_sample_id(tmp_path: Path) -> None:
    token_dir = tmp_path / "tokens"
    importance_dir = tmp_path / "importance"
    sample_ids = ["sample_b", "sample_a"]
    key_mask = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
    token_shard = {
        "schema_version": TOKEN_SHARD_SCHEMA_VERSION,
        "encoder_name": "structured_token_toy",
        "encoder_config": {"num_tokens": 3, "token_dim": 2},
        "created_at": "2026-06-10T00:00:00+00:00",
        "split": "structured_toy",
        "sample_ids": sample_ids,
        "task_texts": ["task b", "task a"],
        "past_tokens": torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
        "future_tokens": torch.ones(2, 3, 2),
        "metadata": [
            {"sample_id": "sample_b", "key_token_indices": [1]},
            {"sample_id": "sample_a", "key_token_indices": [0]},
        ],
        "aux_labels": {"key_token_mask": key_mask},
    }
    importance_shard = {
        "schema_version": IMPORTANCE_SHARD_SCHEMA_VERSION,
        "importance_method": "teacher_token_occlusion",
        "teacher_checkpoint": "teacher.pt",
        "teacher_config": {},
        "source_token_shard": "tokens.pt",
        "created_at": "2026-06-10T00:00:00+00:00",
        "split": "structured_toy",
        "sample_ids": ["sample_a", "sample_b"],
        "task_texts": ["task a", "task b"],
        "importance_scores": torch.tensor([[9.0, 1.0, 0.0], [0.0, 8.0, 0.0]]),
        "importance_scores_norm": torch.tensor([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0]]),
        "base_losses": torch.zeros(2),
        "masked_losses": torch.ones(2, 3),
        "metadata": [{"sample_id": "sample_a"}, {"sample_id": "sample_b"}],
        "mask_config": {"mask_mode": "zero"},
    }
    save_token_shard(token_dir / "tokens_shard_000000.pt", token_shard)
    save_importance_shard(importance_dir / "importance_shard_000000.pt", importance_shard)

    dataset = StudentSelectorDataset(token_dir, importance_dir)

    assert len(dataset) == 2
    assert dataset[0]["sample_id"] == "sample_a"
    assert torch.equal(dataset[0]["past_tokens"], token_shard["past_tokens"][1])
    assert torch.equal(dataset[0]["importance_scores_norm"], importance_shard["importance_scores_norm"][0])
    assert torch.equal(dataset[0]["key_token_mask"], torch.tensor([1.0, 0.0, 0.0]))

    batch = student_selector_collate_fn([dataset[0], dataset[1]])
    assert batch["past_tokens"].shape == (2, 3, 2)
    assert batch["future_tokens"].shape == (2, 3, 2)
    assert batch["importance_scores"].shape == (2, 3)
    assert batch["importance_scores_norm"].shape == (2, 3)
    assert batch["key_token_mask"].shape == (2, 3)


def test_student_selector_dataset_allows_missing_key_mask_when_not_required(tmp_path: Path) -> None:
    token_dir = tmp_path / "tokens"
    importance_dir = tmp_path / "importance"
    token_shard = {
        "schema_version": TOKEN_SHARD_SCHEMA_VERSION,
        "encoder_name": "videomae",
        "encoder_config": {"num_tokens": 4, "token_dim": 2},
        "created_at": "2026-06-11T00:00:00+00:00",
        "split": "real_minimal",
        "sample_ids": ["sample_0"],
        "task_texts": ["task 0"],
        "past_tokens": torch.randn(1, 4, 2),
        "future_tokens": torch.randn(1, 4, 2),
        "metadata": [{"sample_id": "sample_0"}],
    }
    importance_shard = {
        "schema_version": IMPORTANCE_SHARD_SCHEMA_VERSION,
        "importance_method": "teacher_token_occlusion",
        "teacher_checkpoint": "teacher.pt",
        "teacher_config": {},
        "source_token_shard": "tokens.pt",
        "created_at": "2026-06-11T00:00:00+00:00",
        "split": "real_minimal",
        "sample_ids": ["sample_0"],
        "task_texts": ["task 0"],
        "importance_scores": torch.rand(1, 4),
        "importance_scores_norm": torch.rand(1, 4),
        "base_losses": torch.zeros(1),
        "masked_losses": torch.ones(1, 4),
        "metadata": [{"sample_id": "sample_0"}],
        "mask_config": {"mask_mode": "zero"},
    }
    save_token_shard(token_dir / "tokens_shard_000000.pt", token_shard)
    save_importance_shard(importance_dir / "importance_shard_000000.pt", importance_shard)

    dataset = StudentSelectorDataset(token_dir, importance_dir, require_key_token_mask=False)
    batch = student_selector_collate_fn([dataset[0]])

    assert dataset[0]["key_token_mask"] is None
    assert dataset[0]["has_key_token_mask"] is False
    assert batch["key_token_mask"] is None
    assert batch["has_key_token_mask"] is False
