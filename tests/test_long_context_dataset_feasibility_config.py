from pathlib import Path

import yaml


def test_step19_config_forbids_training_and_large_downloads():
    config = yaml.safe_load(Path("configs/long_context_dataset_feasibility_step19.yaml").read_text(encoding="utf-8"))
    assert config["stage"] == "long_context_dataset_feasibility_step19"
    assert config["local_resource_limits"]["no_full_dataset_download"] is True
    assert config["local_resource_limits"]["no_training"] is True
    assert config["local_resource_limits"]["no_model_download"] is True
    assert config["local_resource_limits"]["max_download_gb"] <= 1.0


def test_step19_config_has_required_candidates_and_current_token_guards():
    config = yaml.safe_load(Path("configs/long_context_dataset_feasibility_step19.yaml").read_text(encoding="utf-8"))
    names = {item["name"] for item in config["candidate_datasets"]}
    assert {"BridgeData V2", "DROID"}.issubset(names)
    for item in config["candidate_datasets"]:
        assert item["allow_full_download"] is False
        assert item["allow_tiny_sample_download"] is False
        assert item["max_download_gb"] <= 1.0
    current = config["selection_design"]["current_tokens"]
    assert current["preserve_full_current"] is True
    assert current["drop_current_tokens"] is False
    assert current["train_current_importance"] is False
