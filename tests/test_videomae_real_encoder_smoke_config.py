"""Config-only tests for Step 9C real-video VideoMAE extraction smoke."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_real_video_videomae_extraction_config_is_conservative() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "token_extraction_real_video_videomae.yaml").read_text()
    )

    assert config["dataset"]["max_samples"] <= 8
    assert config["extraction"]["max_samples"] <= 8
    assert config["extraction"]["batch_size"] == 1
    assert config["extraction"]["num_workers"] == 0
    assert config["encoder"]["name"] == "videomae"
    assert config["encoder"]["allow_download"] is False
    assert config["encoder"]["local_files_only"] is True
    assert config["encoder"]["source_policy"] == "windows_local_then_transfer"
    assert config["encoder"]["image_size"] <= 224
    assert config["encoder"]["num_frames"] <= 8
    assert config["fallback"]["allow_dummy_fallback"] is False
