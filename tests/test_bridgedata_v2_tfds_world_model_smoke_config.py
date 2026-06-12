from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/bridgedata_v2_tfds_world_model_smoke_step26.yaml")


def test_step26_config_exists_and_is_bounded():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["stage"] == "bridgedata_v2_tfds_world_model_smoke_step26"
    assert config["sample_limits"]["max_samples"] <= 4
    assert config["context_bottleneck"]["current_tokens_kept_full"] is True
    assert config["context_bottleneck"]["train_current_importance"] is False
    assert config["world_model_smoke"]["no_training"] is True
    assert config["world_model_smoke"]["no_optimizer_step"] is True
    policy_names = {policy["name"] for policy in config["policies"]}
    assert {"current_only", "random_context_topk", "proxy_importance_topk", "full_context_reference"} <= policy_names


def test_step26_config_guards_no_training_download_or_generation():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    guards = config["guards"]
    for key in [
        "no_download",
        "no_new_data_download",
        "no_new_tfds_shard_download",
        "no_model_download",
        "no_token_extraction",
        "no_importance_generation",
        "no_training",
        "no_optimizer_step",
        "no_teacher_training",
        "no_selector_training",
        "no_world_model_training",
        "no_write_data_token_shards",
        "no_write_data_importance_shards",
    ]:
        assert guards[key] is True
    output_text = "\n".join(str(value) for value in config["output"].values())
    assert "runs/bridgedata_v2_tfds_world_model_smoke_step26_v1" in output_text
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text
