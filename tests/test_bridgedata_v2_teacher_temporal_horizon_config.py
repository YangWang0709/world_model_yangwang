from pathlib import Path

import yaml


def test_step31_config_guards_and_outputs_exist():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_teacher_temporal_horizon_diagnosis_step31.yaml").read_text(encoding="utf-8"))
    guards = config["guards"]
    assert guards["no_token_extraction"] is True
    assert guards["no_proxy_importance_regeneration"] is True
    assert guards["no_teacher_label_regeneration"] is True
    assert guards["no_selector_training"] is True
    assert guards["no_current_importance_training"] is True
    assert guards["allow_diagnostic_predictor_training"] is True
    assert "bridgedata_v2_tfds_teacher_temporal_horizon_diagnosis_step31_v1" in "\n".join(config["output"].values())
    assert config["decision"]["context_utility_claim_allowed"] is False
    assert config["decision"]["selector_training_allowed"] is False
    assert config["decision"]["current_importance_training_allowed"] is False
