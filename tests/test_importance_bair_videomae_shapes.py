from __future__ import annotations

import math

import torch

from data.importance_shards import IMPORTANCE_SHARD_SCHEMA_VERSION, save_importance_shard
from eval.eval_importance_bair_videomae_summary import evaluate_bair_importance_root
from models.teacher_world_model import TeacherWorldModel
from scripts.generate_predictive_importance import compute_occlusion_importance


def test_bair_videomae_occlusion_importance_supports_392_tokens():
    torch.manual_seed(7)
    model = TeacherWorldModel(token_dim=768, hidden_dim=16, output_dim=768, num_layers=1)
    past_tokens = torch.randn(1, 392, 768)
    future_tokens = torch.randn(1, 392, 768)

    result = compute_occlusion_importance(
        model,
        past_tokens,
        future_tokens,
        token_chunk_size=32,
        mask_mode="zero",
        mask_value=0.0,
    )

    assert result["base_losses"].shape == (1,)
    assert result["masked_losses"].shape == (1, 392)
    assert result["importance_scores"].shape == (1, 392)
    assert result["importance_scores_norm"].shape == (1, 392)
    for value in result.values():
        assert torch.isfinite(value).all()
        assert not torch.isnan(value).any()


def _write_fake_importance_shard(root, split: str, offset: float) -> None:
    scores = torch.linspace(-0.5 + offset, 0.5 + offset, steps=392).reshape(1, 392)
    normalized = (scores - scores.min()) / (scores.max() - scores.min())
    shard = {
        "schema_version": IMPORTANCE_SHARD_SCHEMA_VERSION,
        "importance_method": "teacher_token_occlusion",
        "teacher_checkpoint": "fake_teacher.pt",
        "teacher_config": {"token_dim": 768},
        "source_token_shard": f"fake_{split}_tokens.pt",
        "created_at": "2026-06-11T00:00:00+00:00",
        "split": split,
        "sample_ids": [f"{split}_sample_0"],
        "task_texts": ["push object"],
        "importance_scores": scores,
        "importance_scores_norm": normalized,
        "base_losses": torch.tensor([1.0 + offset]),
        "masked_losses": torch.ones(1, 392) * (1.0 + offset) + scores,
        "metadata": [{"dataset": "bair_robot_pushing_small"}],
        "mask_config": {"mask_mode": "zero", "mask_value": 0.0},
    }
    save_importance_shard(root / split / "importance_shard_000000.pt", shard)


def test_bair_importance_eval_aggregates_train_test_without_real_data(tmp_path):
    _write_fake_importance_shard(tmp_path, "train", 0.0)
    _write_fake_importance_shard(tmp_path, "test", 0.1)

    summary = evaluate_bair_importance_root(tmp_path)

    assert summary["split"] == "overall"
    assert summary["num_samples"] == 2
    assert summary["num_tokens"] == 392
    assert summary["split_summaries"]["train"]["num_samples"] == 1
    assert summary["split_summaries"]["test"]["num_samples"] == 1
    assert (tmp_path / "train" / "eval_importance_summary.json").exists()
    assert (tmp_path / "test" / "eval_importance_summary.json").exists()
    assert (tmp_path / "eval_importance_summary.json").exists()
    for key in (
        "importance_mean",
        "importance_std",
        "normalized_importance_mean",
        "base_loss_mean",
        "masked_loss_mean",
    ):
        assert math.isfinite(float(summary[key]))
