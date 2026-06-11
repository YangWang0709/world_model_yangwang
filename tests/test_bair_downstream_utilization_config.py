from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/downstream_utilization_bair_1000_128.yaml")


def test_step16_config_points_to_fixed_step15_outputs():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["stage"] == "downstream_utilization_bair_1000_128"
    assert "bair_videomae_1000_128" in config["data"]["train_token_shard_dir"]
    assert "bair_videomae_1000_128" in config["data"]["test_token_shard_dir"]
    assert "bair_videomae_teacher_1000_128" in config["data"]["train_importance_shard_dir"]
    assert "bair_videomae_teacher_1000_128" in config["data"]["test_importance_shard_dir"]
    assert config["selection"]["topk"] == 16
    assert config["phase_b"]["seeds"] == [0, 1, 2]


def test_step16_config_contains_required_variants_and_no_download_or_action_stage():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    variant_names = {variant["name"] for variant in config["phase_a"]["variants"]}
    assert {
        "current_perceiver_like",
        "mean_pool_selected",
        "attention_pool_selected",
        "selector_score_weighted_pool",
        "transformer_encoder_selected",
        "cross_attention_latent_bottleneck",
        "hybrid_learned8_uniform8_perceiver",
        "hybrid_learned8_uniform8_cross_attention",
    }.issubset(variant_names)
    serialized = yaml.safe_dump(config).lower()
    assert "allow_download" not in serialized
    assert "download_bair" not in serialized
    assert "vlm" not in serialized
    assert "dreamer" not in serialized
    assert "td-mpc" not in serialized
    assert "gdpo" not in serialized
    assert "action-conditioned" not in serialized
