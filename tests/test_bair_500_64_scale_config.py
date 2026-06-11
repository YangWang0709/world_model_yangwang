"""Config guards for Step 14 BAIR 500/64 scale validation."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = [
    "bair_500_64_scale_validation.yaml",
    "export_bair_subset_500_64.yaml",
    "token_extraction_bair_videomae_500_64.yaml",
    "train_teacher_bair_videomae_500_64.yaml",
    "generate_importance_bair_videomae_500_64.yaml",
    "train_selector_bair_videomae_500_64_weighted_mse.yaml",
    "train_student_world_model_bair_videomae_500_64_weighted_mse.yaml",
    "baseline_comparison_bair_videomae_500_64.yaml",
]


def _load(name: str) -> dict:
    return yaml.safe_load((PROJECT_ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_step14_all_configs_exist() -> None:
    for name in CONFIG_NAMES:
        assert (PROJECT_ROOT / "configs" / name).exists()


def test_master_config_is_bounded_and_offline() -> None:
    config = _load("bair_500_64_scale_validation.yaml")
    assert config["dataset"]["train_samples"] == 500
    assert config["dataset"]["test_samples"] == 64
    assert config["selector"]["topk"] == 16
    assert config["student_world_model"]["topk"] == 16
    assert config["tokens"]["batch_size"] <= 1
    assert config["tokens"]["num_workers"] == 0
    assert config["importance"]["token_chunk_size"] == 32
    assert config["encoder"]["allow_download"] is False
    assert config["encoder"]["local_files_only"] is True
    assert config["encoder"]["model_name_or_path"].startswith("/home/ubuntu22/tgpawb_world_model/model_cache/")
    assert config["environments"]["tfds_export_env"] != "env_isaaclab"
    assert "4090" not in yaml.safe_dump(config).lower()
    assert "vlm" not in yaml.safe_dump(config).lower()
    assert "dreamer" not in yaml.safe_dump(config).lower()
    assert "gdpo" not in yaml.safe_dump(config).lower()
    assert "action-conditioned" not in yaml.safe_dump(config).lower()


def test_step14_leaf_configs_point_to_500_64_artifacts() -> None:
    export_cfg = _load("export_bair_subset_500_64.yaml")
    token_cfg = _load("token_extraction_bair_videomae_500_64.yaml")
    teacher_cfg = _load("train_teacher_bair_videomae_500_64.yaml")
    importance_cfg = _load("generate_importance_bair_videomae_500_64.yaml")
    selector_cfg = _load("train_selector_bair_videomae_500_64_weighted_mse.yaml")
    student_cfg = _load("train_student_world_model_bair_videomae_500_64_weighted_mse.yaml")
    baseline_cfg = _load("baseline_comparison_bair_videomae_500_64.yaml")

    assert export_cfg["subset"]["train_max_episodes"] == 500
    assert export_cfg["subset"]["test_max_episodes"] == 64
    assert "bair_robot_pushing_small_subset_500_64" in token_cfg["dataset"]["root"]
    assert token_cfg["fallback"]["allow_dummy_fallback"] is False
    assert teacher_cfg["training"]["max_steps"] == 800
    assert importance_cfg["importance"]["batch_size"] == 1
    assert selector_cfg["loss_variants"] == [
        {"name": "weighted_mse_alpha2", "type": "weighted_mse", "seed": 0, "alpha": 2.0, "topk": 16}
    ]
    assert student_cfg["selector"]["checkpoint"].endswith("student_selector_step_000800.pt")
    assert baseline_cfg["training"]["repeat_random_seeds"]["random_k"] == [0, 1, 2]
    assert set(baseline_cfg["selection"]["policies"]) == {
        "random_k",
        "uniform_k",
        "teacher_importance_topk",
        "learned_selector",
    }
    for cfg in (token_cfg, teacher_cfg, importance_cfg, selector_cfg, student_cfg, baseline_cfg):
        text = yaml.safe_dump(cfg)
        assert "bair_videomae_smoke" not in text
        assert "selector_ablation_bair_videomae_smoke" not in text
