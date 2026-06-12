from pathlib import Path

import yaml


def test_step32_config_forbids_downloads_and_forbidden_training():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_longer_horizon_step32.yaml").read_text(encoding="utf-8"))
    guards = config["guards"]
    for key in [
        "no_new_data_download",
        "no_new_tfds_shard_download",
        "no_raw_zip_download",
        "no_full_tfds_download",
        "no_droid_download",
        "no_model_download",
        "no_videomae_training",
        "no_selector_training",
        "no_current_importance_training",
        "no_action_input",
        "no_vlm",
        "no_rl",
        "no_write_data_token_shards",
        "no_write_data_importance_shards",
    ]:
        assert guards[key] is True
    assert config["clip_export"]["use_action_as_input"] is False
    assert config["clip_export"]["use_language_as_input"] is False
    assert config["clip_export"]["use_goal_image_as_input"] is False


def test_step32_scripts_do_not_call_downloaders_or_selector_training():
    script_text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in [
            "scripts/prepare_bridgedata_v2_tfds_longer_horizon_windows_step32.py",
            "scripts/extract_bridgedata_v2_tfds_longer_horizon_tokens_step32.py",
            "scripts/generate_bridgedata_v2_tfds_longer_horizon_importance_step32.py",
            "scripts/train_eval_bridgedata_v2_tfds_longer_horizon_step32.py",
        ]
    )
    forbidden_calls = [
        "download_bridgedata",
        "download_bair",
        "download_videomae",
        "UnifiedPredictiveImportanceSelector(",
        "train_current_importance",
        "action_conditioned",
    ]
    for forbidden in forbidden_calls[:3]:
        assert forbidden not in script_text
    assert "UnifiedPredictiveImportanceSelector(" not in script_text
    assert "current_importance_training_performed" in script_text
    assert "action_conditioned_world_model_training_performed" in script_text

