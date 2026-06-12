from pathlib import Path

import yaml


def test_step23_5_config_forbids_download_and_training():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_field_resolver_step23_5.yaml").read_text(encoding="utf-8"))
    assert config["guards"]["no_download"] is True
    assert config["guards"]["no_new_tfds_shard_download"] is True
    assert config["guards"]["no_raw_zip_download"] is True
    assert config["guards"]["no_full_tfds_download"] is True
    assert config["guards"]["no_training"] is True
    assert config["guards"]["no_token_extraction"] is True
    assert config["guards"]["no_importance_generation"] is True
    assert config["guards"]["keep_tensorflow_out_of_env_isaaclab"] is True
    assert config["input"]["redownload_allowed"] is False
