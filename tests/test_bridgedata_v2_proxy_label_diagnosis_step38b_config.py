from pathlib import Path

import yaml

from data.bridgedata_v2_proxy_label_diagnosis_step38b import STAGE, load_step38b_config


CONFIG = Path("configs/bridgedata_v2_tfds_proxy_label_diagnosis_step38b.yaml")


def test_step38b_config_has_safe_defaults():
    config = load_step38b_config(CONFIG)

    assert config["stage"] == STAGE
    assert config["data"]["context_frames"] == 16
    assert config["data"]["spatial_tokens"] == 392
    assert config["data"]["use_action_as_input"] is False
    assert config["data"]["use_language_as_input"] is False
    assert config["data"]["load_future_tokens_for_diagnosis"] is False
    assert config["future_gates"]["selector_training_allowed"] is False
    assert config["future_gates"]["final_selector_training_allowed"] is False
    assert config["future_gates"]["current_importance_training_allowed"] is False
    assert config["future_gates"]["context_utility_claim_allowed"] is False
    assert config["future_gates"]["label_patch_detail_learnable_allowed"] is False
    assert config["guards"]["no_training"] is True
    assert config["guards"]["no_optimizer_step"] is True
    assert config["guards"]["no_checkpoint_save"] is True
    assert config["guards"]["no_tensorflow_import_in_env_isaaclab"] is True
    assert config["guards"]["no_tensorflow_datasets_import_in_env_isaaclab"] is True


def test_step38b_config_outputs_only_ignored_run_dir():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output_text = "\n".join(str(value) for value in config["output"].values())

    assert "runs/bridgedata_v2_tfds_proxy_label_diagnosis_step38b_v1" in output_text
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text
    assert "checkpoints" not in output_text
