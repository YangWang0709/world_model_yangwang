from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "generate_importance_real_video_videomae_smoke.yaml"


def test_real_videomae_importance_config_is_bounded():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert "real_video_videomae_smoke" in config["data"]["token_shard_dir"]
    assert "teacher_real_video_videomae_smoke_v1" in config["teacher"]["checkpoint"]
    assert int(config["importance"]["batch_size"]) <= 1
    assert int(config["importance"]["token_chunk_size"]) <= 32
    assert int(config["data"]["max_samples"]) <= 8
    assert "download" not in config
    assert "encoder" not in config
