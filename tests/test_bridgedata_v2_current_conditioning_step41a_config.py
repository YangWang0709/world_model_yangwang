from pathlib import Path

from data.bridgedata_v2_current_conditioning_dataset_step41a import load_step41a_config


def test_step41a_config_declares_bounded_diagnosis_and_false_future_gates():
    config = load_step41a_config("configs/bridgedata_v2_tfds_current_conditioning_diagnosis_step41a.yaml")

    assert config["stage"] == "bridgedata_v2_tfds_current_conditioning_diagnosis_step41a"
    assert config["training"]["bounded_smoke_training_only"] is True
    assert config["training"]["save_checkpoint"] is False
    assert config["training"]["save_state_dict"] is False
    assert config["model"]["freeze_videomae"] is True
    assert config["model"]["train_video_encoder"] is False
    assert config["data"]["use_action_as_input"] is False
    assert config["data"]["use_language_as_input"] is False
    assert config["data"]["load_future_tokens_for_selector"] is False
    assert config["splits"]["strict_shard_aware_splits"] is True
    assert config["current_conditioning_variants"]["primary_variant"] == "current_coarse_spatial_query_attention"
    assert config["future_gates"]["downstream_selector_use_allowed"] is False
    assert config["future_gates"]["final_selector_training_allowed"] is False
    assert config["future_gates"]["current_importance_training_allowed"] is False
    assert config["future_gates"]["context_utility_claim_allowed"] is False
    assert Path(config["output"]["run_dir"]).name.endswith("step41a_v1")
