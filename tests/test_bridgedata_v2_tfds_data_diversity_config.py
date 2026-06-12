from pathlib import Path

import yaml


def test_step33b_config_exists_and_allows_only_one_extra_shard():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_data_diversity_step33b.yaml").read_text(encoding="utf-8"))
    assert config["stage"] == "bridgedata_v2_tfds_data_diversity_step33b"
    shard = config["second_shard"]
    assert shard["allow_one_new_tfds_shard_download"] is True
    assert shard["max_new_shards_to_download"] == 1
    assert shard["no_raw_zip"] is True
    assert shard["no_full_tfds"] is True
    assert shard["no_model_download"] is True
    assert config["guards"]["no_selector_training"] is True
    assert config["guards"]["no_current_importance_training"] is True
    assert config["decision"]["selector_training_allowed"] is False
    assert config["decision"]["current_importance_training_allowed"] is False


def test_step33b_config_uses_gap0_delta_frame_repeat():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_data_diversity_step33b.yaml").read_text(encoding="utf-8"))
    assert config["window"]["horizon_gap"] == 0
    assert config["window"]["target_variant"] == "future_delta_last_minus_current"
    assert config["token_extraction"]["tokenization_mode"] == "frame_repeat_baseline"
    assert config["proxy_importance"]["train_current_importance"] is False
    assert config["clip_export"]["use_action_as_input"] is False
    assert config["clip_export"]["use_language_as_input"] is False
