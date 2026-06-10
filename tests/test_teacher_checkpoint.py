"""Tests for teacher checkpoint helpers."""

from pathlib import Path

from models.teacher_world_model import TeacherWorldModel
from training.teacher_trainer import load_checkpoint, save_checkpoint


def test_teacher_checkpoint_save_load(tmp_path: Path) -> None:
    model_config = {
        "token_dim": 768,
        "hidden_dim": 128,
        "output_dim": 768,
        "num_layers": 2,
        "dropout": 0.0,
        "pool": "mean",
    }
    model = TeacherWorldModel(**model_config)
    checkpoint_path = save_checkpoint(
        tmp_path / "teacher.pt",
        model=model,
        optimizer=None,
        step=3,
        model_config=model_config,
        metrics_summary={"best_loss": 0.1},
    )
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint["step"] == 3
    assert checkpoint["model_config"] == model_config
    assert "model_state_dict" in checkpoint
    assert len(checkpoint["model_state_dict"]) > 0
    assert sum(tensor.numel() for tensor in checkpoint["model_state_dict"].values()) > 0

