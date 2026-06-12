from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/bridgedata_v2_tfds_context_sanity_step28.yaml")


def test_step28_config_exists_and_has_stage():
    assert CONFIG_PATH.exists()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["stage"] == "bridgedata_v2_tfds_context_sanity_step28"
    assert config["analysis"]["use_summary_files_only_by_default"] is True
    assert config["analysis"]["load_token_artifacts_for_stats"] is False
    assert config["analysis"]["load_importance_artifacts_for_stats"] is False


def test_step28_config_guardrails_forbid_training_download_and_shard_writes():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    guards = config["guards"]
    for key in [
        "no_download",
        "no_new_data_download",
        "no_new_tfds_shard_download",
        "no_model_download",
        "no_training",
        "no_optimizer_step",
        "no_token_extraction",
        "no_importance_generation",
        "no_current_importance_training",
        "no_write_data_token_shards",
        "no_write_data_importance_shards",
        "keep_tensorflow_out_of_env_isaaclab",
    ]:
        assert guards[key] is True
    output_text = "\n".join(str(value) for value in config["output"].values())
    assert "runs/bridgedata_v2_tfds_context_sanity_step28_v1" in output_text
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text

