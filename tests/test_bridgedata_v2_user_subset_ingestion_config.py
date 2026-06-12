from pathlib import Path

import yaml


def test_step22_config_has_user_subset_ingestion_guards():
    config = yaml.safe_load(Path("configs/bridgedata_v2_user_subset_ingestion_step22.yaml").read_text(encoding="utf-8"))
    assert config["stage"] == "bridgedata_v2_user_subset_ingestion_step22"
    assert config["dataset"]["no_download"] is True
    assert config["dataset"]["full_download_allowed"] is False
    assert config["dataset"]["large_download_allowed"] is False
    guards = config["guards"]
    assert guards["no_download"] is True
    assert guards["no_training"] is True
    assert guards["no_model_download"] is True
    assert guards["no_token_extraction"] is True
    assert guards["no_importance_generation"] is True
    assert config["metadata_policy"]["use_action_as_input"] is False
    assert config["metadata_policy"]["use_language_as_input"] is False
    assert config["metadata_policy"]["use_goal_image_as_input"] is False
    assert config["window"]["context_len"] == 16
    assert config["window"]["current_len"] == 4
    assert config["window"]["future_len"] == 4
