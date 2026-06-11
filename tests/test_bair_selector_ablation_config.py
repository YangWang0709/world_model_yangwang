from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path("configs/selector_ablation_bair_videomae_smoke.yaml")


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def test_bair_selector_ablation_config_is_bounded_and_offline() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert "bair_videomae_smoke/train" in config["data"]["train_token_shard_dir"]
    assert "bair_videomae_smoke/test" in config["data"]["test_token_shard_dir"]
    assert "bair_videomae_teacher_smoke/train" in config["data"]["train_importance_shard_dir"]
    assert "bair_videomae_teacher_smoke/test" in config["data"]["test_importance_shard_dir"]
    assert config["selection"]["topk"] == 16
    assert config["selector_training"]["batch_size"] <= 4
    assert config["selector_training"]["max_steps"] <= 300
    assert config["downstream_student_world_model"]["max_steps"] <= 500
    assert config["selector_training"]["num_workers"] == 0
    assert config["downstream_student_world_model"]["num_workers"] == 0
    assert config["data"]["max_train_samples"] == 100
    assert config["data"]["max_test_samples"] == 16

    loss_types = {item["type"] for item in config["loss_variants"]}
    assert {
        "mse_only",
        "weighted_mse",
        "mse_plus_pairwise_rank",
        "topk_bce",
        "hybrid_weighted_mse_rank_bce",
    }.issubset(loss_types)

    keys = _walk_keys(config)
    assert "download" not in keys
    assert "tfds" not in keys
    assert "videomae_checkpoint" not in keys
