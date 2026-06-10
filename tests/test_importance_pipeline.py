from __future__ import annotations

import torch

from data.importance_shards import load_importance_shard, validate_importance_shard
from data.token_shards import TOKEN_SHARD_SCHEMA_VERSION, save_token_shard
from models.teacher_world_model import TeacherWorldModel
from scripts.generate_predictive_importance import generate_predictive_importance
from training.teacher_trainer import save_checkpoint


def test_generate_predictive_importance_from_temporary_inputs(tmp_path):
    token_dir = tmp_path / "tokens"
    output_dir = tmp_path / "importance"
    checkpoint_path = tmp_path / "teacher.pt"
    token_dim = 4
    model_config = {
        "token_dim": token_dim,
        "hidden_dim": 8,
        "output_dim": token_dim,
        "num_layers": 2,
        "dropout": 0.0,
        "pool": "mean",
    }
    model = TeacherWorldModel(**model_config)
    save_checkpoint(checkpoint_path, model, optimizer=None, step=1, model_config=model_config)

    save_token_shard(
        token_dir / "tokens_shard_000000.pt",
        {
            "schema_version": TOKEN_SHARD_SCHEMA_VERSION,
            "encoder_name": "dummy",
            "encoder_config": {"token_dim": token_dim},
            "created_at": "2026-06-10T00:00:00+00:00",
            "split": "toy",
            "sample_ids": ["s0", "s1"],
            "task_texts": ["task 0", "task 1"],
            "past_tokens": torch.randn(2, 3, token_dim),
            "future_tokens": torch.randn(2, 3, token_dim),
            "metadata": [{"index": 0}, {"index": 1}],
        },
    )

    summary = generate_predictive_importance(
        {
            "data": {
                "token_shard_dir": str(token_dir),
                "shard_glob": "tokens_shard_*.pt",
                "split": "toy",
            },
            "teacher": {"checkpoint": str(checkpoint_path), "config": model_config},
            "importance": {
                "method": "teacher_token_occlusion",
                "mask_mode": "zero",
                "mask_value": 0.0,
                "token_chunk_size": 2,
                "device": "cpu",
                "clamp_negative_importance": False,
                "normalize": "minmax_per_sample",
            },
            "output": {
                "output_dir": str(output_dir),
                "summary_path": str(output_dir / "importance_summary.json"),
                "overwrite": True,
            },
        }
    )

    assert summary["num_importance_shards"] == 1
    shard = load_importance_shard(output_dir / "importance_shard_000000.pt")
    assert validate_importance_shard(shard)
    assert shard["importance_scores"].shape == (2, 3)
