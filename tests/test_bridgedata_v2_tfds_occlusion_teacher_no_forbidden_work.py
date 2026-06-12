from pathlib import Path

import yaml


def test_step30b_scripts_do_not_call_token_or_proxy_regeneration_or_forbidden_training():
    script_paths = [
        Path("scripts/train_bridgedata_v2_tfds_occlusion_teacher_step30b.py"),
        Path("scripts/generate_bridgedata_v2_tfds_occlusion_teacher_labels_step30b.py"),
        Path("scripts/evaluate_bridgedata_v2_tfds_teacher_topk_step30b.py"),
        Path("scripts/smoke_test_bridgedata_v2_tfds_occlusion_teacher_step30b.py"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in script_paths)
    forbidden = [
        "extract_bridgedata_v2_tfds_tokens_from_config",
        "extract_step30a_window_diversity_tokens",
        "generate_bridgedata_v2_tfds_importance_from_config",
        "generate_step30a_window_diversity_importance",
        "download_bridgedata",
        "download_bair",
        "download_videomae_checkpoint",
        "UnifiedPredictiveImportanceSelector",
        "from scripts.train_selector",
        "from training.train_selector",
    ]
    assert [word for word in forbidden if word in text] == []


def test_step30b_guards_keep_forbidden_work_disabled():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_occlusion_teacher_step30b.yaml").read_text())
    guards = config["guards"]
    assert guards["no_token_extraction"] is True
    assert guards["no_proxy_importance_regeneration"] is True
    assert guards["no_new_tfds_shard_download"] is True
    assert guards["no_model_download"] is True
    assert guards["no_videomae_training"] is True
    assert guards["no_context_teacher_large_training"] is True
    assert guards["no_selector_training"] is True
    assert guards["no_current_importance_training"] is True
    assert guards["no_write_data_token_shards"] is True
    assert guards["no_write_data_importance_shards"] is True
