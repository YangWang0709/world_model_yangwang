import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_proxy_label_diagnosis_step38b import evaluate_step38b_proxy_label_diagnosis


def test_step38b_eval_writes_reports_and_keeps_flags_false(tmp_path: Path):
    config = yaml.safe_load(
        Path("configs/bridgedata_v2_tfds_proxy_label_diagnosis_step38b.yaml").read_text(encoding="utf-8")
    )
    run = tmp_path / "run"
    config["output"]["run_dir"] = str(run)
    config["output"]["diagnosis_summary_json"] = str(run / "diagnosis_summary.json")
    config["output"]["label_decomposition_json"] = str(run / "label_decomposition.json")
    config["output"]["cross_shard_consistency_json"] = str(run / "cross_shard_consistency.json")
    config["output"]["gate_decision_json"] = str(run / "step38b_gate_decision.json")
    config["output"]["eval_json"] = str(run / "step38b_eval.json")
    config["output"]["eval_md"] = str(run / "step38b_eval.md")
    run.mkdir()
    aggregate = {
        "num_rows": 2,
        "residual_energy_ratio_mean": 0.2,
        "temporal_energy_ratio_mean": 0.8,
        "temporal_broadcast_r2_like_mean": 0.7,
        "per_frame_spatial_entropy_mean_mean": 0.95,
        "top256_mass_ratio_mean": 0.2,
        "top256_residual_mass_ratio_mean": 0.2,
        "temporal_component_dominates": True,
        "spatial_residual_learnable_evidence": False,
    }
    _write(
        config["output"]["diagnosis_summary_json"],
        {
            "stage": config["stage"],
            "safe_stop": False,
            "diagnosis_performed": True,
            "num_samples": 2,
            "all_samples": aggregate,
            "training_performed": False,
            "optimizer_step_performed": False,
            "checkpoint_saved": False,
            "state_dict_saved": False,
            "data_token_shards_written": False,
            "data_importance_shards_written": False,
            "new_dataset_download_performed": False,
            "new_model_download_performed": False,
        },
    )
    _write(
        config["output"]["label_decomposition_json"],
        {
            "stage": config["stage"],
            "num_rows": 2,
            "rows": [],
            "training_performed": False,
            "optimizer_step_performed": False,
            "checkpoint_saved": False,
            "state_dict_saved": False,
        },
    )
    _write(
        config["output"]["cross_shard_consistency_json"],
        {
            "stage": config["stage"],
            "computed": True,
            "cross_shard_residual_cosine": 0.1,
            "cross_shard_residual_l1": 0.01,
            "cross_shard_residual_consistent": False,
        },
    )
    _write(
        config["output"]["gate_decision_json"],
        {
            "stage": config["stage"],
            "temporal_component_dominates": True,
            "spatial_residual_learnable_evidence": False,
            "spatial_residual_cross_shard_consistent": False,
            "label_patch_detail_learnable_allowed": False,
            "selector_training_allowed": False,
            "final_selector_training_allowed": False,
            "current_importance_training_allowed": False,
            "context_utility_claim_allowed": False,
            "optimizer_step_performed": False,
            "checkpoint_saved": False,
            "state_dict_saved": False,
            "data_token_shards_written": False,
            "data_importance_shards_written": False,
            "new_dataset_download_performed": False,
            "new_model_download_performed": False,
            "recommended_step39": {
                "name": "label redesign toward temporal-only or temporal+broadcast/coarse supervision",
                "scope": "redesign proxy labels before any downstream selector use",
                "reason": "fake diffuse residual",
            },
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = evaluate_step38b_proxy_label_diagnosis(config_path)

    assert result["pass"] is True
    assert result["selector_training_allowed"] is False
    assert result["final_selector_training_allowed"] is False
    assert result["current_importance_training_allowed"] is False
    assert result["context_utility_claim_allowed"] is False
    assert Path(config["output"]["eval_json"]).exists()
    assert Path("docs/STEP38B_PROXY_LABEL_LEARNABILITY_DIAGNOSIS.md").exists()
    assert "must not claim context utility" in Path("docs/STEP38B_PROXY_LABEL_LEARNABILITY_DIAGNOSIS.md").read_text(
        encoding="utf-8"
    )


def _write(path: str, payload: dict):
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
