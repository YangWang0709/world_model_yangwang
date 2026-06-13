from pathlib import Path

import yaml


CONFIG = Path("configs/bridgedata_v2_tfds_proxy_temporal_selector_train_step35.yaml")


def test_step35_config_exists_and_keeps_decision_flags_false():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["stage"] == "bridgedata_v2_tfds_proxy_temporal_selector_train_step35"
    assert config["selector"]["module"] == "ProxyTemporalSelectorHead"
    assert config["selector"]["temporal_only"] is True
    assert config["selector"]["patch_level_selector"] is False
    assert config["selector"]["final_selector"] is False
    assert config["future_gates"]["selector_training_allowed"] is False
    assert config["future_gates"]["final_selector_training_allowed"] is False
    assert config["future_gates"]["current_importance_training_allowed"] is False
    assert config["future_gates"]["context_utility_claim_allowed"] is False


def test_step35_config_forbids_forbidden_work_and_checkpoints():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    guards = config["guards"]
    for key in [
        "no_raw_zip_download",
        "no_full_tfds_download",
        "no_droid_download",
        "no_open_x_embodiment_download",
        "no_new_tfds_shard_download",
        "no_new_pretrained_model_download",
        "no_checkpoint_download",
        "no_image_or_video_download",
        "no_compressed_archive_download",
        "no_extra_cloud_resources",
        "no_bair_redownload",
        "no_videomae_token_reextraction",
        "no_proxy_importance_regeneration",
        "use_existing_local_artifacts_only",
        "use_existing_step33b_artifacts_only",
        "use_existing_step34_scaffold",
        "no_videomae_training",
        "no_final_selector_training",
        "no_patch_level_selector_training",
        "no_current_importance_training",
        "no_world_model_training",
        "no_downstream_task_training",
        "no_action_input",
        "no_language_input",
        "no_write_data_token_shards",
        "no_write_data_importance_shards",
        "no_checkpoint_save",
    ]:
        assert guards[key] is True
    assert config["training"]["save_checkpoint"] is False
    assert config["training"]["save_state_dict"] is False
    assert config["training"]["allowed_optimizer_scopes"] == ["proxy_temporal_selector_head_only"]
