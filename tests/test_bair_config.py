"""Config tests for Step 11A BAIR Robot Pushing small."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _walk_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        out: list[Any] = []
        for key, child in value.items():
            out.append(key)
            out.extend(_walk_values(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(_walk_values(child))
        return out
    return [value]


def test_bair_download_config_is_bounded() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "bair_robot_pushing_small.yaml").read_text())

    assert config["dataset"]["tfds_name"] == "bair_robot_pushing_small"
    assert config["dataset"]["tfds_version"] == "2.0.0"
    assert config["dataset"]["download"] is True
    assert config["resource_limits"]["min_disk_free_gib"] >= 80
    assert config["resource_limits"]["max_retries"] <= 1


def test_bair_subset_config_is_tiny_and_model_free() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "bair_robot_pushing_subset.yaml").read_text())
    subset = config["subset"]
    all_values = _walk_values(config)

    assert subset["train_max_episodes"] <= 100
    assert subset["test_max_episodes"] <= 16
    assert subset["total_frames"] == 8
    assert subset["past_len"] == 4
    assert subset["future_len"] == 4
    assert subset["image_size"] == 224
    assert "model_name_or_path" not in all_values
    assert "allow_download" not in all_values
    assert "encoder" not in all_values
