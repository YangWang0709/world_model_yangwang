from pathlib import Path

import yaml


def test_step30a_config_exists_and_preserves_guards():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_window_diversity_step30a.yaml").read_text())
    assert config["stage"] == "bridgedata_v2_tfds_window_diversity_step30a"
    assert config["window_diversity"]["selected_counts_to_run"] == [64]
    assert config["guards"]["no_new_tfds_shard_download"] is True
    assert config["guards"]["no_model_download"] is True
    assert config["guards"]["allow_limited_token_extraction_from_existing_shard"] is True
    assert config["guards"]["allow_limited_proxy_importance_generation"] is True
    assert config["guards"]["no_selector_training"] is True
    assert config["guards"]["no_current_importance_training"] is True
    assert config["tiny_trainval"]["save_checkpoint"] is False
    assert config["stability_decision"]["context_utility_claim_allowed"] is False


def test_step30a_output_paths_stay_in_ignored_run_dir():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_window_diversity_step30a.yaml").read_text())
    output_text = "\n".join(str(value) for value in config["output"].values())
    assert "runs/bridgedata_v2_tfds_window_diversity_step30a_v1" in output_text
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text
    assert "checkpoints" not in output_text
