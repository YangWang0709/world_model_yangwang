from pathlib import Path

import yaml


def test_step30b_config_exists_and_preserves_training_boundary():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_occlusion_teacher_step30b.yaml").read_text())
    assert config["stage"] == "bridgedata_v2_tfds_occlusion_teacher_step30b"
    assert config["guards"]["no_token_extraction"] is True
    assert config["guards"]["no_proxy_importance_regeneration"] is True
    assert config["guards"]["allow_trained_predictor_teacher_training"] is True
    assert config["guards"]["allow_teacher_occlusion_label_generation"] is True
    assert config["guards"]["no_selector_training"] is True
    assert config["guards"]["no_current_importance_training"] is True
    assert config["teacher_training"]["save_teacher_checkpoint"] is False
    assert config["teacher_topk_utility"]["context_utility_claim_allowed"] is False


def test_step30b_outputs_stay_in_ignored_run_dir():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_occlusion_teacher_step30b.yaml").read_text())
    output_text = "\n".join(str(value) for value in config["output"].values())
    assert "runs/bridgedata_v2_tfds_occlusion_teacher_step30b_v1" in output_text
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text
    assert "checkpoints" not in output_text
