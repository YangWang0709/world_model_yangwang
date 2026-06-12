from pathlib import Path

import yaml


def test_step27_tiny_overfit_config_has_strict_bounds():
    config = yaml.safe_load(
        Path("configs/bridgedata_v2_tfds_world_model_tiny_overfit_step27.yaml").read_text(encoding="utf-8")
    )
    assert config["stage"] == "bridgedata_v2_tfds_world_model_tiny_overfit_step27"
    assert config["sample_limits"]["max_samples"] <= 4
    assert config["sample_limits"]["batch_size"] <= 4
    assert config["tiny_overfit"]["enabled"] is True
    assert config["tiny_overfit"]["train_steps"] <= 300
    assert config["tiny_overfit"]["save_checkpoint"] is False
    assert config["tiny_overfit"]["save_state_dict"] is False
    assert config["context_bottleneck"]["current_tokens_kept_full"] is True
    assert config["context_bottleneck"]["train_current_importance"] is False
    assert config["guards"]["allow_tiny_world_model_training"] is True
    assert config["guards"]["no_videomae_training"] is True
    assert config["guards"]["no_teacher_training"] is True
    assert config["guards"]["no_selector_training"] is True
    assert config["guards"]["no_current_importance_training"] is True
    assert config["guards"]["no_token_extraction"] is True
    assert config["guards"]["no_importance_generation"] is True
    assert config["guards"]["no_write_data_token_shards"] is True
    assert config["guards"]["no_write_data_importance_shards"] is True
    assert config["guards"]["no_download"] is True
    assert config["guards"]["no_model_download"] is True
    assert [policy["name"] for policy in config["policies"]] == [
        "current_only",
        "random_context_topk",
        "proxy_importance_topk",
        "full_context_reference",
    ]


def test_step27_output_paths_are_ignored_run_dir_only():
    config = yaml.safe_load(
        Path("configs/bridgedata_v2_tfds_world_model_tiny_overfit_step27.yaml").read_text(encoding="utf-8")
    )
    output_text = "\n".join(str(value) for value in config["output"].values())
    assert "runs/bridgedata_v2_tfds_world_model_tiny_overfit_step27_v1" in output_text
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text
    assert "checkpoints" not in output_text
