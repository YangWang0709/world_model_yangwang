from pathlib import Path

import yaml


CONFIG = Path("configs/bridgedata_v2_tfds_longer_horizon_step32.yaml")


def test_step32_config_guards_and_outputs_exist():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    guards = config["guards"]
    assert guards["no_new_tfds_shard_download"] is True
    assert guards["no_model_download"] is True
    assert guards["allow_limited_token_extraction_from_existing_shard"] is True
    assert guards["allow_limited_proxy_importance_generation"] is True
    assert guards["allow_tiny_diagnostic_predictor_training"] is True
    assert guards["no_selector_training"] is True
    assert guards["no_current_importance_training"] is True
    assert guards["keep_tensorflow_out_of_env_isaaclab"] is True
    assert config["decision"]["context_utility_claim_allowed"] is False
    assert config["decision"]["selector_training_allowed"] is False
    assert config["decision"]["current_importance_training_allowed"] is False
    assert config["tiny_trainval"]["save_checkpoint"] is False
    assert "bridgedata_v2_tfds_longer_horizon_step32_v1" in "\n".join(config["output"].values())


def test_step32_config_window_targets_and_metadata_only_inputs():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["window"]["context_len"] == 16
    assert config["window"]["current_len"] == 4
    assert config["window"]["future_len"] == 4
    assert config["window"]["horizon_gaps"] == [0, 4, 8, 12]
    assert config["window"]["required_horizon_gaps"] == [0, 4, 8]
    assert config["window"]["action_language_goal_as_metadata_only"] is True
    assert config["targets"]["primary_target"] == "future_delta_last_minus_current"
    assert "future_delta_mean_minus_current" in config["targets"]["target_variants"]

