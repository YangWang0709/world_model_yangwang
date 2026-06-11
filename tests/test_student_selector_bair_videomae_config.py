from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path("configs/train_student_selector_bair_videomae_smoke.yaml")


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_walk_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_walk_keys(child))
        return keys
    return set()


def test_bair_student_selector_config_is_bounded_to_existing_artifacts():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["data"]["train_token_shard_dir"].endswith("bair_videomae_smoke/train")
    assert config["data"]["test_token_shard_dir"].endswith("bair_videomae_smoke/test")
    assert config["data"]["train_importance_shard_dir"].endswith("bair_videomae_teacher_smoke/train")
    assert config["data"]["test_importance_shard_dir"].endswith("bair_videomae_teacher_smoke/test")
    assert config["data"]["require_key_token_mask"] is False
    assert config["training"]["batch_size"] <= 4
    assert config["training"]["max_steps"] <= 500
    assert config["training"]["num_workers"] == 0
    assert config["training"]["topk"] <= 32
    assert config["data"]["max_train_samples"] <= 100
    assert config["data"]["max_test_samples"] <= 16

    forbidden_keys = {
        "download",
        "download_url",
        "dataset_url",
        "hf_model_id",
        "model_name",
        "model_name_or_path",
        "pretrained_model_name_or_path",
    }
    assert not (_walk_keys(config) & forbidden_keys)
