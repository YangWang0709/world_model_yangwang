from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/bridgedata_v2_tfds_token_extraction_step24.yaml")


def test_step24_token_extraction_config_is_guarded():
    assert CONFIG_PATH.exists()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["resolved_fields"]["image_field"] == "steps/observation/image_0"
    assert int(config["dry_run_limits"]["max_windows"]) <= 4
    assert config["guards"]["no_new_data_download"] is True
    assert config["guards"]["no_new_tfds_shard_download"] is True
    assert config["guards"]["no_model_download"] is True
    assert config["guards"]["no_training"] is True
    assert config["guards"]["no_importance_generation"] is True
    assert config["guards"]["no_large_token_shards"] is True
    assert config["guards"]["keep_tensorflow_out_of_env_isaaclab"] is True
    assert config["clip_export"]["use_action_as_input"] is False
    assert config["clip_export"]["use_language_as_input"] is False
    assert config["clip_export"]["use_goal_image_as_input"] is False
    assert "runs/bridgedata_v2_tfds_token_extraction_step24_v1" in config["output"]["token_smoke_dir"]
    assert "data/token_shards" not in config["output"]["token_smoke_dir"]
    assert "data/importance_shards" not in config["output"]["token_smoke_dir"]
