from pathlib import Path

import yaml


def test_context_selector_oracle_gap_config():
    config_path = Path("configs/context_selector_oracle_gap_bair_1000_128.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data = config["data"]
    assert "data/context_token_shards/bair_context_videomae_1000_128" in data["context_token_shard_dir_train"]
    assert "data/context_token_shards/bair_context_videomae_1000_128" in data["context_token_shard_dir_test"]
    assert "data/context_importance_shards/bair_context_teacher_1000_128" in data["context_importance_shard_dir_train"]
    assert "data/context_importance_shards/bair_context_teacher_1000_128" in data["context_importance_shard_dir_test"]
    assert config["selection"]["context_topk"] == 32
    assert config["selection"]["current_tokens_dropped"] is False
    assert config["selection"]["train_current_importance"] is False

    variants = {item["name"]: item for item in config["selector_variants"]}
    assert "weighted_mse_alpha2" in variants
    assert "weighted_mse_alpha5" in variants
    assert "topk_bce" in variants
    assert "pairwise_rank_w0p1" in variants
    assert "hybrid_weighted_mse_rank_bce" in variants
    assert "weighted_mse_alpha2_no_current_condition" in variants
    assert "weighted_mse_alpha2_no_temporal_pos" in variants
    assert "temporal_block_balanced_topk" in variants
    assert variants["temporal_block_balanced_topk"]["topk_per_block"] == 4

    serialized = config_path.read_text(encoding="utf-8").lower()
    for forbidden in ("vlm", "dreamer", "td-mpc", "gdpo", "action-conditioned", "action_conditioned", "model_download"):
        assert forbidden not in serialized
