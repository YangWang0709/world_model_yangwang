from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_student_world_model_bair_videomae_smoke.yaml"


def test_bair_videomae_student_world_model_config_is_bounded() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data_cfg = config["data"]

    assert "bair_videomae_smoke/train" in data_cfg["train_token_shard_dir"]
    assert "bair_videomae_smoke/test" in data_cfg["test_token_shard_dir"]
    assert "bair_videomae_teacher_smoke/train" in data_cfg["train_importance_shard_dir"]
    assert "bair_videomae_teacher_smoke/test" in data_cfg["test_importance_shard_dir"]
    assert "student_selector_bair_videomae_smoke_v1" in config["selector"]["checkpoint"]
    assert "teacher_bair_videomae_smoke_v1" in config["teacher_reference"]["checkpoint"]
    assert data_cfg["require_key_token_mask"] is False
    assert int(data_cfg["max_train_samples"]) <= 100
    assert int(data_cfg["max_test_samples"]) <= 16
    assert config["selector"]["freeze"] is True
    assert int(config["selection"]["topk"]) == 16
    assert int(config["training"]["batch_size"]) <= 4
    assert int(config["training"]["max_steps"]) <= 500
    assert int(config["training"]["num_workers"]) == 0
    assert "download" not in config
    assert "bair_download" not in config
