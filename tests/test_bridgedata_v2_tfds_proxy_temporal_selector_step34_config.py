from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/bridgedata_v2_tfds_proxy_temporal_selector_step34.yaml")


def test_step34_config_is_conservative_by_default():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["stage"] == "bridgedata_v2_tfds_proxy_temporal_selector_step34"
    assert config["decision"]["may_prepare_proxy_supervised_selector_training"] is True
    assert config["decision"]["selector_training_allowed"] is False
    assert config["decision"]["current_importance_training_allowed"] is False
    assert config["decision"]["context_utility_claim_allowed"] is False
    assert config["decision"]["final_selector_training_allowed"] is False
    assert config["decision"]["downstream_task_improvement_evidence_allowed"] is False


def test_step34_config_forbids_downloads_and_keeps_videomae_frozen():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    guards = config["guards"]
    for key in [
        "no_raw_zip_download",
        "no_full_tfds_download",
        "no_droid_download",
        "no_new_pretrained_model_download",
        "no_checkpoint_download",
        "no_image_or_video_download",
        "no_compressed_archive_download",
        "no_extra_cloud_resources",
        "use_existing_local_tfds_shards_only",
        "use_existing_local_videomae_only",
        "use_existing_step33b_artifacts_only",
        "no_final_selector_training",
        "no_current_importance_training",
        "no_videomae_training",
        "no_downstream_task_training",
        "no_action_input",
        "no_write_data_token_shards",
        "no_write_data_importance_shards",
        "no_checkpoint_save",
    ]:
        assert guards[key] is True
    selector = config["selector_scaffold"]
    assert selector["freeze_videomae"] is True
    assert selector["detach_token_inputs"] is True
    assert selector["train_video_encoder"] is False
    assert selector["train_final_selector"] is False
    assert selector["train_current_importance"] is False
    assert selector["optimizer_step_allowed"] is False
    assert selector["save_checkpoint"] is False
    assert selector["save_state_dict"] is False


def test_step34_split_plan_requires_shard_awareness():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    split_plan = config["split_plan"]
    assert split_plan["strict_shard_aware_splits"] is True
    assert split_plan["within_shard_validation"] is True
    assert split_plan["cross_shard_validation"] is True
    assert split_plan["mixed_shard_validation"] is True
    assert split_plan["train_val_sample_id_disjoint"] is True
    assert split_plan["train_val_trajectory_disjoint_if_possible"] is True
    assert split_plan["do_not_use_downstream_task_improvement_as_evidence"] is True
