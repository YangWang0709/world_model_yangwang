from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/bridgedata_v2_tfds_importance_step25.yaml")


def test_step25_importance_config_exists_and_is_bounded():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["stage"] == "bridgedata_v2_tfds_importance_step25"
    assert config["importance"]["method"] == "proxy_token_mse_dryrun"
    assert config["importance"]["target"] == "context_tokens_only"
    assert config["importance"]["current_tokens_kept_full"] is True
    assert config["importance"]["train_current_importance"] is False
    assert config["dry_run_limits"]["max_samples"] <= 4
    assert config["token_shapes"]["context"] == [16, 392, 768]
    assert config["token_shapes"]["current"] == [4, 392, 768]
    assert config["token_shapes"]["future"] == [4, 392, 768]


def test_step25_importance_config_guards_no_download_training_or_token_extraction():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    guards = config["guards"]
    for key in [
        "no_download",
        "no_new_data_download",
        "no_new_tfds_shard_download",
        "no_model_download",
        "no_training",
        "no_teacher_training",
        "no_selector_training",
        "no_world_model_training",
        "no_token_extraction",
        "no_write_data_importance_shards",
        "no_write_data_token_shards",
        "no_current_importance",
    ]:
        assert guards[key] is True
    output_text = "\n".join(str(value) for value in config["output"].values())
    assert "runs/bridgedata_v2_tfds_importance_step25_v1" in output_text
    assert "data/importance_shards" not in output_text
    assert "data/token_shards" not in output_text
