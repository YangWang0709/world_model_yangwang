from pathlib import Path

import yaml


def test_step20_config_forbids_download_training_tokens_and_importance():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tiny_window_builder_step20.yaml").read_text(encoding="utf-8"))
    dataset = config["dataset"]
    guards = config["guards"]
    policy = config["metadata_policy"]
    assert config["stage"] == "bridgedata_v2_tiny_window_builder_step20"
    assert dataset["full_download_allowed"] is False
    assert dataset["large_download_allowed"] is False
    assert dataset["max_download_gb"] <= 1.0
    assert guards["no_training"] is True
    assert guards["no_model_download"] is True
    assert guards["no_token_extraction"] is True
    assert guards["no_importance_generation"] is True
    assert policy["use_action_as_input"] is False
    assert policy["use_language_as_input"] is False
    assert policy["use_goal_image_as_input"] is False


def test_step20_config_window_spec_matches_requested_scope():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tiny_window_builder_step20.yaml").read_text(encoding="utf-8"))
    window = config["window"]
    assert window["context_len"] == 16
    assert window["current_len"] == 4
    assert window["future_len"] == 4
    assert window["min_trajectory_len"] == 24
    assert config["dataset"]["allow_missing_local_subset"] is True
    assert config["dataset"]["use_fake_manifest_if_missing"] is True
