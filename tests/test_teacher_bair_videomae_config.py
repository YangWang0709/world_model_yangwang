"""Config tests for Step 11C Teacher training on BAIR VideoMAE tokens."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_teacher_bair_videomae_config_is_bounded_and_offline() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "train_teacher_bair_videomae_smoke.yaml").read_text()
    )

    assert "bair_videomae_smoke/train" in config["data"]["train_token_shard_dir"]
    assert "bair_videomae_smoke/test" in config["data"]["test_token_shard_dir"]
    assert config["data"]["max_train_samples"] <= 100
    assert config["data"]["max_test_samples"] <= 16
    assert config["training"]["batch_size"] <= 4
    assert config["training"]["max_steps"] <= 300
    assert config["training"]["num_workers"] == 0
    assert config["model"]["token_dim"] == 768
    assert "encoder" not in config
    assert "download" not in config["data"]
    assert config["output"]["run_name"] == "teacher_bair_videomae_smoke_v1"
