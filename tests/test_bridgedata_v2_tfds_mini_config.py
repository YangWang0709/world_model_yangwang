from pathlib import Path

import yaml


def test_step23_config_has_strict_tfds_mini_guards():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_mini_shard_step23.yaml").read_text(encoding="utf-8"))
    assert config["stage"] == "bridgedata_v2_tfds_mini_shard_step23"
    assert config["dataset"]["raw_zip_download_allowed"] is False
    assert config["dataset"]["full_tfds_download_allowed"] is False
    assert config["dataset"]["mini_shard_download_allowed"] is True
    assert config["download_policy"]["max_download_gb"] <= 1.0
    assert config["download_policy"]["max_download_bytes"] <= 1073741824
    assert config["guards"]["no_training"] is True
    assert config["guards"]["no_token_extraction"] is True
    assert config["guards"]["no_importance_generation"] is True
    assert config["guards"]["keep_tensorflow_out_of_env_isaaclab"] is True
    assert config["window"]["context_len"] == 16
    assert config["window"]["current_len"] == 4
    assert config["window"]["future_len"] == 4
