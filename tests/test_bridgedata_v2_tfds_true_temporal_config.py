from pathlib import Path

import yaml


CONFIG = Path("configs/bridgedata_v2_tfds_true_temporal_step33a.yaml")


def test_step33a_config_guards_outputs_and_local_only_inputs():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    guards = config["guards"]
    assert guards["no_new_tfds_shard_download"] is True
    assert guards["no_model_download"] is True
    assert guards["allow_limited_true_temporal_token_extraction_from_existing_shard"] is True
    assert guards["allow_limited_proxy_importance_generation"] is True
    assert guards["no_videomae_training"] is True
    assert guards["no_selector_training"] is True
    assert guards["no_current_importance_training"] is True
    assert guards["no_write_data_token_shards"] is True
    assert guards["no_write_data_importance_shards"] is True
    assert config["true_temporal_token_extraction"]["local_files_only"] is True
    assert config["true_temporal_token_extraction"]["no_model_download"] is True
    assert config["tiny_trainval"]["save_checkpoint"] is False
    assert config["tiny_trainval"]["save_state_dict"] is False
    assert "bridgedata_v2_tfds_true_temporal_step33a_v1" in "\n".join(config["output"].values())


def test_step33a_config_compares_gap0_delta_target_without_selector_training():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["selection"]["horizon_gap"] == 0
    assert config["selection"]["selected_windows_source"] == "step32_gap0"
    assert config["comparison"]["primary_target"] == "future_delta_last_minus_current"
    assert "frame_repeat_baseline_step32" in config["comparison"]["representation_modes"]
    assert "true_temporal_clip_step33a" in config["comparison"]["representation_modes"]
    assert config["decision"]["context_utility_claim_allowed"] is False
    assert config["decision"]["selector_training_allowed"] is False
    assert config["decision"]["current_importance_training_allowed"] is False
