from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/bridgedata_v2_tfds_trainval_context_utility_step29.yaml")


def test_step29_config_exists_and_targets_existing_shard_run_dir():
    assert CONFIG_PATH.exists()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["stage"] == "bridgedata_v2_tfds_trainval_context_utility_step29"
    assert config["window_selection"]["target_num_windows_primary"] == 32
    assert config["split"]["min_val_windows"] == 8
    output_text = "\n".join(str(value) for value in config["output"].values())
    assert "runs/bridgedata_v2_tfds_trainval_context_utility_step29_v1" in output_text
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text


def test_step29_config_guardrails():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    guards = config["guards"]
    assert guards["no_new_tfds_shard_download"] is True
    assert guards["no_model_download"] is True
    assert guards["allow_limited_token_extraction_from_existing_shard"] is True
    assert guards["allow_limited_proxy_importance_generation"] is True
    assert guards["allow_tiny_world_model_training"] is True
    assert guards["no_selector_training"] is True
    assert guards["no_current_importance_training"] is True
    assert config["tiny_trainval"]["save_checkpoint"] is False

