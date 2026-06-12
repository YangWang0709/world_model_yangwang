from pathlib import Path

import yaml


def test_step29_scripts_do_not_reference_downloaders_or_forbidden_training_entrypoints():
    script_paths = [
        Path("scripts/prepare_bridgedata_v2_tfds_32_64_windows.py"),
        Path("scripts/extract_bridgedata_v2_tfds_32_64_tokens.py"),
        Path("scripts/generate_bridgedata_v2_tfds_32_64_importance.py"),
        Path("scripts/train_eval_bridgedata_v2_tfds_context_utility.py"),
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


def test_step29_outputs_avoid_data_shard_dirs_and_checkpoints():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_trainval_context_utility_step29.yaml").read_text())
    output_text = "\n".join(str(value) for value in config["output"].values())
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text
    assert "checkpoints" not in output_text
    assert config["tiny_trainval"]["save_checkpoint"] is False
    assert config["guards"]["no_videomae_training"] is True
    assert config["guards"]["no_teacher_training"] is True
    assert config["guards"]["no_selector_training"] is True
    assert config["guards"]["no_current_importance_training"] is True
