"""Unit tests for Step 11C BAIR Teacher training without real BAIR shards."""

from __future__ import annotations

from pathlib import Path

import torch

from data.token_shards import save_token_shard
from eval.eval_teacher_prediction import evaluate_teacher
from models.teacher_world_model import TeacherWorldModel
from training.losses import future_latent_mse
from training.teacher_trainer import run_teacher_training, target_from_future_tokens


def _write_token_shard(path: Path, split: str, num_samples: int, num_tokens: int, token_dim: int) -> None:
    shard = {
        "schema_version": "0.1.0",
        "encoder_name": "videomae",
        "encoder_config": {
            "requested_encoder": "videomae",
            "actual_encoder": "videomae",
            "used_fallback": False,
            "num_tokens": num_tokens,
            "token_dim": token_dim,
        },
        "created_at": "2026-06-11T00:00:00+00:00",
        "split": split,
        "sample_ids": [f"bair_{split}_{index:06d}" for index in range(num_samples)],
        "task_texts": ["predict robot pushing future visual dynamics"] * num_samples,
        "past_tokens": torch.randn(num_samples, num_tokens, token_dim),
        "future_tokens": torch.randn(num_samples, num_tokens, token_dim),
        "metadata": [
            {"source": "bair_robot_pushing_small", "split": split, "sample_id": f"bair_{split}_{index:06d}"}
            for index in range(num_samples)
        ],
    }
    save_token_shard(path, shard)


def test_teacher_forward_backward_with_bair_videomae_token_shape() -> None:
    batch_size = 2
    num_tokens = 392
    token_dim = 768
    model = TeacherWorldModel(token_dim=token_dim, hidden_dim=128, output_dim=token_dim, num_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    past_tokens = torch.randn(batch_size, num_tokens, token_dim)
    future_tokens = torch.randn(batch_size, num_tokens, token_dim)

    pred = model(past_tokens)
    target = target_from_future_tokens(future_tokens)
    assert list(pred.shape) == [batch_size, token_dim]
    assert list(target.shape) == [batch_size, token_dim]

    loss = future_latent_mse(pred, target)
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def test_teacher_trainer_and_eval_use_train_test_dirs(tmp_path: Path) -> None:
    train_dir = tmp_path / "tokens" / "train"
    test_dir = tmp_path / "tokens" / "test"
    _write_token_shard(train_dir / "tokens_shard_000000.pt", split="train", num_samples=3, num_tokens=3, token_dim=8)
    _write_token_shard(test_dir / "tokens_shard_000000.pt", split="test", num_samples=2, num_tokens=3, token_dim=8)
    run_root = tmp_path / "runs"
    config = {
        "seed": 7,
        "data": {
            "train_token_shard_dir": str(train_dir),
            "test_token_shard_dir": str(test_dir),
            "shard_glob": "tokens_shard_*.pt",
            "split_train": "train",
            "split_test": "test",
            "max_train_samples": 3,
            "max_test_samples": 2,
        },
        "model": {
            "token_dim": 8,
            "hidden_dim": 16,
            "output_dim": 8,
            "num_layers": 2,
            "dropout": 0.0,
            "pool": "mean",
        },
        "training": {
            "device": "cpu",
            "batch_size": 2,
            "max_steps": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "num_workers": 0,
            "log_every": 0,
            "checkpoint_every": 2,
            "max_grad_norm": 1.0,
        },
        "output": {
            "run_root": str(run_root),
            "run_name": "teacher_bair_unit",
            "save_checkpoint": True,
            "save_metrics": True,
        },
    }

    train_summary = run_teacher_training(config)
    eval_summary = evaluate_teacher(config, train_summary["checkpoint_path"])

    assert train_summary["train_token_shard_dir"] == str(train_dir)
    assert train_summary["test_token_shard_dir"] == str(test_dir)
    assert train_summary["train_num_samples"] == 3
    assert train_summary["test_num_samples"] == 2
    assert train_summary["num_tokens"] == 3
    assert train_summary["token_dim"] == 8
    assert train_summary["dataset"] == "bair_robot_pushing_small"
    assert eval_summary["token_shard_dir"] == str(test_dir)
    assert eval_summary["split"] == "test"
    assert eval_summary["num_samples"] == 2
    assert eval_summary["dataset"] == "bair_robot_pushing_small"
    assert torch.isfinite(torch.tensor(eval_summary["eval_mse"]))
