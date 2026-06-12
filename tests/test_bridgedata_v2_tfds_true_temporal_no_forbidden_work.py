from pathlib import Path

import yaml


def test_step33a_config_forbids_downloads_and_forbidden_training():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_true_temporal_step33a.yaml").read_text(encoding="utf-8"))
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
    assert config["proxy_importance"]["train_current_importance"] is False
    assert config["decision"]["selector_training_allowed"] is False
    assert config["decision"]["current_importance_training_allowed"] is False


def test_step33a_scripts_do_not_call_downloaders_or_selector_training():
    script_text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in [
            "scripts/extract_bridgedata_v2_tfds_true_temporal_tokens_step33a.py",
            "scripts/generate_bridgedata_v2_tfds_true_temporal_importance_step33a.py",
            "scripts/train_eval_bridgedata_v2_tfds_true_temporal_step33a.py",
            "scripts/smoke_test_bridgedata_v2_tfds_true_temporal_step33a.py",
        ]
    )
    for forbidden in [
        "download_bridgedata",
        "download_bair",
        "download_videomae",
        "UnifiedPredictiveImportanceSelector(",
        "data/token_shards",
        "data/importance_shards",
    ]:
        assert forbidden not in script_text
    assert "videomae_training_performed" in script_text
    assert "selector_training_performed" in script_text
    assert "current_importance_training_performed" in script_text
