from pathlib import Path

import yaml


def test_step30a_scripts_do_not_reference_downloaders_or_forbidden_training_entrypoints():
    script_paths = [
        Path("scripts/prepare_bridgedata_v2_tfds_window_diversity_step30a.py"),
        Path("scripts/extract_bridgedata_v2_tfds_window_diversity_tokens_step30a.py"),
        Path("scripts/generate_bridgedata_v2_tfds_window_diversity_importance_step30a.py"),
        Path("scripts/train_eval_bridgedata_v2_tfds_window_diversity_step30a.py"),
        Path("scripts/smoke_test_bridgedata_v2_tfds_window_diversity_step30a.py"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in script_paths)
    forbidden = [
        "download_bridgedata",
        "download_bair",
        "download_videomae_checkpoint",
        "UnifiedPredictiveImportanceSelector",
        "from scripts.train_teacher",
        "from scripts.train_selector",
        "from training.train_teacher",
        "from training.train_selector",
        "UnifiedPredictiveImportanceSelector(",
    ]
    assert [word for word in forbidden if word in text] == []


def test_step30a_guards_keep_forbidden_work_disabled():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_window_diversity_step30a.yaml").read_text())
    guards = config["guards"]
    assert guards["no_new_tfds_shard_download"] is True
    assert guards["no_model_download"] is True
    assert guards["no_videomae_training"] is True
    assert guards["no_teacher_training"] is True
    assert guards["no_selector_training"] is True
    assert guards["no_current_importance_training"] is True
    assert guards["no_write_data_token_shards"] is True
    assert guards["no_write_data_importance_shards"] is True
    assert config["tiny_trainval"]["save_checkpoint"] is False
