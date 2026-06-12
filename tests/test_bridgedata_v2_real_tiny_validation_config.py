from pathlib import Path

import yaml


def test_step21_config_has_strict_download_and_training_guards():
    config = yaml.safe_load(Path("configs/bridgedata_v2_real_tiny_validation_step21.yaml").read_text(encoding="utf-8"))
    dataset = config["dataset"]
    guards = config["guards"]
    assert config["stage"] == "bridgedata_v2_real_tiny_validation_step21"
    assert dataset["max_download_gb"] <= 1.0
    assert dataset["max_download_bytes"] <= 1073741824
    assert dataset["require_content_length_before_download"] is True
    assert guards["no_training"] is True
    assert guards["no_model_download"] is True
    assert guards["no_token_extraction"] is True
    assert guards["no_importance_generation"] is True
    assert guards["no_action_input"] is True
    assert guards["no_droid_download"] is True
    assert config["metadata_policy"]["use_action_as_input"] is False
    assert config["metadata_policy"]["use_language_as_input"] is False
    assert config["metadata_policy"]["use_goal_image_as_input"] is False
