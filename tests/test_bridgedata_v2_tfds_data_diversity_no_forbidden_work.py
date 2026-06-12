from pathlib import Path

import yaml


def test_step33b_config_forbids_forbidden_work():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_data_diversity_step33b.yaml").read_text(encoding="utf-8"))
    guards = config["guards"]
    for key in [
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
    assert guards["max_new_shards_to_download"] == 1
    assert config["tiny_trainval"]["save_checkpoint"] is False


def test_step33b_scripts_do_not_train_forbidden_modules_or_write_data_shards():
    script_text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in [
            "scripts/prepare_bridgedata_v2_tfds_data_diversity_step33b.py",
            "scripts/extract_bridgedata_v2_tfds_data_diversity_tokens_step33b.py",
            "scripts/generate_bridgedata_v2_tfds_data_diversity_importance_step33b.py",
            "scripts/train_eval_bridgedata_v2_tfds_data_diversity_step33b.py",
            "scripts/smoke_test_bridgedata_v2_tfds_data_diversity_step33b.py",
        ]
    )
    for forbidden in [
        "UnifiedPredictiveImportanceSelector(",
        "train_current_importance=True",
        "data/token_shards",
        "data/importance_shards",
        "download_bair",
        "download_videomae",
    ]:
        assert forbidden not in script_text
    assert "downloaded_new_shard_count" in script_text
    assert "videomae_training_performed" in script_text
    assert "selector_training_performed" in script_text
    assert "current_importance_training_performed" in script_text
