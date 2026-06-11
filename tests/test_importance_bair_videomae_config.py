from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path("configs/generate_importance_bair_videomae_smoke.yaml")


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


def test_bair_videomae_importance_config_is_bounded_to_existing_artifacts():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["data"]["train_token_shard_dir"].endswith("bair_videomae_smoke/train")
    assert config["data"]["test_token_shard_dir"].endswith("bair_videomae_smoke/test")
    assert "teacher_bair_videomae_smoke_v1" in config["teacher"]["checkpoint"]
    assert config["importance"]["batch_size"] <= 1
    assert config["importance"]["token_chunk_size"] <= 32
    assert config["data"]["max_train_samples"] <= 100
    assert config["data"]["max_test_samples"] <= 16
    assert config["output"]["output_root"].endswith("data/importance_shards/bair_videomae_teacher_smoke")
    assert config["output"]["overwrite"] is True

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
