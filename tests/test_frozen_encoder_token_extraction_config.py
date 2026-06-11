"""Config tests for conservative Step 9B frozen-token extraction."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_video_encoder_smoke_config_is_conservative() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "frozen_video_encoder_smoke.yaml").read_text())

    assert config["encoder"]["allow_download"] is False
    assert config["encoder"]["local_files_only"] is True
    assert config["resource_limits"]["batch_size"] == 1
    assert config["resource_limits"]["num_workers"] == 0
    assert config["resource_limits"]["max_samples"] <= 16
    assert config["fallback"]["allow_dummy_fallback"] is True


def test_real_video_frozen_extraction_config_is_conservative() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "token_extraction_real_video_frozen_encoder.yaml").read_text()
    )

    assert config["encoder"]["allow_download"] is False
    assert config["encoder"]["local_files_only"] is True
    assert config["dataset"]["max_samples"] <= 16
    assert config["extraction"]["batch_size"] == 1
    assert config["extraction"]["num_workers"] == 0
    assert config["fallback"]["allow_dummy_fallback"] is True
