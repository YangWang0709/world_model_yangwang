"""Config-only tests for Step 9C VideoMAE download smoke."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_videomae_real_encoder_smoke_config_is_conservative() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "videomae_real_video_smoke.yaml").read_text())
    project_root = Path(config["project_root"])
    model_path = Path(config["encoder"]["model_name_or_path"])
    cache_dir = Path(config["encoder"]["cache_dir"])

    assert config["encoder"]["name"] == "videomae"
    assert config["encoder"]["allow_download"] is False
    assert config["encoder"]["local_files_only"] is True
    assert config["encoder"]["source_policy"] == "windows_local_then_transfer"
    assert str(model_path).startswith(str(project_root / "model_cache"))
    assert str(cache_dir).startswith(str(project_root / "model_cache"))
    assert config["resource_limits"]["batch_size"] == 1
    assert config["resource_limits"]["max_samples"] <= 4
    assert config["resource_limits"]["image_size"] <= 224
    assert config["resource_limits"]["clip_len"] <= 8
    assert config["fallback"]["allow_dummy_fallback"] is False
