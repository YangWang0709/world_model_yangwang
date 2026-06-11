"""Config tests for Step 10A Teacher training on real VideoMAE tokens."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_teacher_real_videomae_config_is_tiny_and_offline() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "train_teacher_real_video_videomae_smoke.yaml").read_text()
    )

    assert "real_video_videomae_smoke" in config["data"]["token_shard_dir"]
    assert config["training"]["batch_size"] <= 2
    assert config["training"]["max_steps"] <= 100
    assert config["training"]["num_workers"] == 0
    assert config["model"]["token_dim"] == 768
    assert "encoder" not in config
    assert config["output"]["run_name"] == "teacher_real_video_videomae_smoke_v1"
