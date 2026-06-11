from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_student_world_model_real_video_videomae_smoke.yaml"


def test_real_videomae_student_world_model_config_is_bounded() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert "real_video_videomae_smoke" in config["data"]["token_shard_dir"]
    assert "real_video_videomae_teacher_smoke" in config["data"]["importance_shard_dir"]
    assert "student_selector_real_video_videomae_smoke_v1" in config["selector"]["checkpoint"]
    assert config["data"]["require_key_token_mask"] is False
    assert int(config["data"]["max_samples"]) <= 8
    assert config["selector"]["freeze"] is True
    assert int(config["selection"]["topk"]) == 16
    assert int(config["training"]["batch_size"]) <= 2
    assert int(config["training"]["max_steps"]) <= 500
    assert int(config["training"]["num_workers"]) == 0
    assert "teacher_real_video_videomae_smoke_v1" in config["teacher_reference"]["checkpoint"]
