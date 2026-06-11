from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_student_selector_real_video_videomae_smoke.yaml"


def test_real_videomae_student_selector_config_is_bounded():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert "real_video_videomae_smoke" in config["data"]["token_shard_dir"]
    assert "real_video_videomae_teacher_smoke" in config["data"]["importance_shard_dir"]
    assert config["data"]["require_key_token_mask"] is False
    assert int(config["data"]["max_samples"]) <= 8
    assert int(config["training"]["batch_size"]) <= 2
    assert int(config["training"]["max_steps"]) <= 300
    assert int(config["training"]["num_workers"]) == 0
    assert int(config["training"]["topk"]) <= 32
    assert float(config["training"]["ranking_loss_weight"]) == 0.0
