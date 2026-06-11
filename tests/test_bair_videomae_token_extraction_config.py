"""Config guards for Step 11B BAIR VideoMAE token extraction."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_bair_videomae_token_extraction_config_is_bounded() -> None:
    config_path = Path("configs/token_extraction_bair_videomae_smoke.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["dataset"]["name"] == "bair_robot_pushing_small_subset"
    assert config["dataset"]["max_train_samples"] <= 100
    assert config["dataset"]["max_test_samples"] <= 16
    assert config["extraction"]["batch_size"] == 1
    assert config["extraction"]["num_workers"] == 0
    assert config["encoder"]["name"] == "videomae"
    assert config["encoder"]["allow_download"] is False
    assert config["encoder"]["local_files_only"] is True
    assert config["fallback"]["allow_dummy_fallback"] is False
    assert config["extraction"]["output_root"].endswith("data/token_shards/bair_videomae_smoke")
