from pathlib import Path

import yaml


def test_step31_config_forbids_downloads_extraction_and_forbidden_training():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_teacher_temporal_horizon_diagnosis_step31.yaml").read_text(encoding="utf-8"))
    guards = config["guards"]
    for key in [
        "no_new_tfds_shard_download",
        "no_raw_zip_download",
        "no_full_tfds_download",
        "no_droid_download",
        "no_model_download",
        "no_token_extraction",
        "no_proxy_importance_regeneration",
        "no_teacher_label_regeneration",
        "no_videomae_training",
        "no_selector_training",
        "no_current_importance_training",
        "no_write_data_token_shards",
        "no_write_data_importance_shards",
    ]:
        assert guards[key] is True
    output_text = "\n".join(str(value) for value in config["output"].values())
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text
    assert "checkpoints" not in output_text
