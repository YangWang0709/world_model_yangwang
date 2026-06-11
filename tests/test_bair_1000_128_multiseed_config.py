"""Config guards for Step 15 BAIR 1000/128 multi-seed validation."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = [
    "bair_1000_128_multiseed_validation.yaml",
    "export_bair_subset_1000_128.yaml",
    "token_extraction_bair_videomae_1000_128.yaml",
    "train_teacher_bair_videomae_1000_128.yaml",
    "generate_importance_bair_videomae_1000_128.yaml",
    "train_selector_bair_videomae_1000_128_weighted_mse.yaml",
    "train_student_world_model_bair_videomae_1000_128_weighted_mse.yaml",
    "baseline_comparison_bair_videomae_1000_128_multiseed.yaml",
]


def _load(name: str) -> dict:
    return yaml.safe_load((PROJECT_ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_step15_all_configs_exist() -> None:
    for name in CONFIG_NAMES:
        assert (PROJECT_ROOT / "configs" / name).exists()


def test_master_config_is_bounded_offline_and_multiseed() -> None:
    config = _load("bair_1000_128_multiseed_validation.yaml")
    dumped = yaml.safe_dump(config).lower()
    assert config["dataset"]["train_samples"] == 1000
    assert config["dataset"]["test_samples"] == 128
    assert config["selector"]["topk"] == 16
    assert config["student_world_model"]["topk"] == 16
    assert config["selector"]["seeds"] == [0, 1, 2]
    assert config["baseline"]["random_k_seeds"] == [0, 1, 2]
    assert config["baseline"]["learned_selector_seeds"] == [0, 1, 2]
    assert config["tokens"]["batch_size"] <= 1
    assert config["tokens"]["num_workers"] == 0
    assert config["teacher"]["batch_size"] <= 4
    assert config["student_world_model"]["batch_size"] <= 4
    assert config["importance"]["token_chunk_size"] == 32
    assert config["encoder"]["allow_download"] is False
    assert config["encoder"]["local_files_only"] is True
    assert config["encoder"]["model_name_or_path"].startswith("/home/ubuntu22/tgpawb_world_model/model_cache/")
    assert config["environments"]["tfds_export_env"] != "env_isaaclab"
    assert "aws" not in dumped
    assert "azure" not in dumped
    assert "vlm" not in dumped
    assert "dreamer" not in dumped
    assert "td-mpc" not in dumped
    assert "gdpo" not in dumped
    assert "action-conditioned" not in dumped


def test_step15_leaf_configs_point_to_1000_128_artifacts() -> None:
    export_cfg = _load("export_bair_subset_1000_128.yaml")
    token_cfg = _load("token_extraction_bair_videomae_1000_128.yaml")
    teacher_cfg = _load("train_teacher_bair_videomae_1000_128.yaml")
    importance_cfg = _load("generate_importance_bair_videomae_1000_128.yaml")
    selector_cfg = _load("train_selector_bair_videomae_1000_128_weighted_mse.yaml")
    student_cfg = _load("train_student_world_model_bair_videomae_1000_128_weighted_mse.yaml")
    baseline_cfg = _load("baseline_comparison_bair_videomae_1000_128_multiseed.yaml")

    assert export_cfg["subset"]["train_max_episodes"] == 1000
    assert export_cfg["subset"]["test_max_episodes"] == 128
    assert "bair_robot_pushing_small_subset_1000_128" in token_cfg["dataset"]["root"]
    assert token_cfg["fallback"]["allow_dummy_fallback"] is False
    assert teacher_cfg["training"]["max_steps"] == 1200
    assert importance_cfg["importance"]["batch_size"] == 1
    assert importance_cfg["importance"]["token_chunk_size"] == 32
    assert [variant["seed"] for variant in selector_cfg["loss_variants"]] == [0, 1, 2]
    assert {variant["type"] for variant in selector_cfg["loss_variants"]} == {"weighted_mse"}
    assert {float(variant["alpha"]) for variant in selector_cfg["loss_variants"]} == {2.0}
    assert student_cfg["selector"]["checkpoint"].endswith("student_selector_step_001000.pt")
    assert student_cfg["training"]["max_steps"] == 1200
    assert baseline_cfg["training"]["repeat_random_seeds"]["random_k"] == [0, 1, 2]
    assert baseline_cfg["training"]["repeat_random_seeds"]["learned_selector"] == [0, 1, 2]
    assert set(baseline_cfg["selection"]["policies"]) == {
        "random_k",
        "uniform_k",
        "teacher_importance_topk",
        "learned_selector",
    }
    for cfg in (token_cfg, teacher_cfg, importance_cfg, selector_cfg, student_cfg, baseline_cfg):
        text = yaml.safe_dump(cfg)
        assert "500_64" not in text
        assert "bair_videomae_smoke" not in text
        assert "selector_ablation_bair_videomae_smoke" not in text
