from pathlib import Path

import yaml

from data.bridgedata_v2_redesigned_label_selector_dataset_step40a import STAGE, load_step40a_config


CONFIG = Path("configs/bridgedata_v2_tfds_redesigned_label_selector_step40a.yaml")


def test_step40a_config_has_safe_bounded_smoke_defaults():
    config = load_step40a_config(CONFIG)

    assert config["stage"] == STAGE
    assert config["label"]["variant"] == "global_spatial_prior_removed_residual"
    assert config["label"]["build_global_spatial_prior_from_train_split_only"] is True
    assert config["label"]["do_not_save_label_tensor_artifacts"] is True
    assert config["data"]["use_action_as_input"] is False
    assert config["data"]["use_language_as_input"] is False
    assert config["data"]["load_future_tokens_for_selector"] is False
    assert config["training"]["bounded_smoke_training_only"] is True
    assert config["training"]["save_checkpoint"] is False
    assert config["training"]["save_state_dict"] is False
    assert config["future_gates"]["downstream_selector_use_allowed"] is False
    assert config["future_gates"]["final_selector_training_allowed"] is False
    assert config["future_gates"]["current_importance_training_allowed"] is False
    assert config["future_gates"]["context_utility_claim_allowed"] is False
    assert config["guards"]["no_checkpoint_save"] is True
    assert config["guards"]["no_tensorflow_import_in_env_isaaclab"] is True
    assert config["guards"]["no_tensorflow_datasets_import_in_env_isaaclab"] is True


def test_step40a_config_outputs_only_ignored_run_dir():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output_text = "\n".join(str(value) for value in config["output"].values())

    assert "runs/bridgedata_v2_tfds_redesigned_label_selector_step40a_v1" in output_text
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text
    assert "checkpoints" not in output_text
