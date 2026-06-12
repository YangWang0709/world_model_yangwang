import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_tfds_importance import evaluate_step25_importance


def _base_config(tmp_path):
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_importance_step25.yaml").read_text(encoding="utf-8"))
    config["output"]["importance_summary_json"] = str(tmp_path / "summary.json")
    config["output"]["importance_manifest_jsonl"] = str(tmp_path / "manifest.jsonl")
    config["output"]["eval_json"] = str(tmp_path / "eval.json")
    config["output"]["eval_md"] = str(tmp_path / "eval.md")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, path


def _success_summary():
    return {
        "stage": "bridgedata_v2_tfds_importance_step25",
        "importance_generation_performed": True,
        "safe_stop": False,
        "num_samples": 1,
        "num_importance_artifacts": 1,
        "method": "proxy_token_mse_dryrun",
        "context_importance_shape_example": [16, 392],
        "temporal_importance_shape_example": [16],
        "spatial_importance_shape_example": [392],
        "importance_norm_min": 0.0,
        "importance_norm_max": 1.0,
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "download_performed": False,
        "model_download_performed": False,
        "training_performed": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "world_model_training_performed": False,
        "token_extraction_performed": False,
        "large_importance_shards_generated": False,
        "data_importance_shards_written": False,
        "data_token_shards_written": False,
        "safety_gate_pass": True,
    }


def _manifest_record():
    return {
        "sample_id": "sample",
        "trajectory_id": "traj",
        "importance_artifact_path": "/tmp/sample.pt",
        "method": "proxy_token_mse_dryrun",
        "context_importance_shape": [16, 392],
        "temporal_importance_shape": [16],
        "spatial_importance_shape": [392],
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
    }


def test_success_report_passes_and_recommends_step26(tmp_path):
    config, config_path = _base_config(tmp_path)
    Path(config["output"]["importance_summary_json"]).write_text(json.dumps(_success_summary()), encoding="utf-8")
    Path(config["output"]["importance_manifest_jsonl"]).write_text(
        json.dumps(_manifest_record()) + "\n",
        encoding="utf-8",
    )
    result = evaluate_step25_importance(config_path)
    assert result["pass"] is True
    assert result["safe_stop"] is False
    assert result["label_quality_note"] == "proxy dry-run only; not final teacher label"
    assert result["recommended_step26"]["condition"] == "importance dry-run succeeded"


def test_safe_stop_does_not_pretend_to_pass(tmp_path):
    config, config_path = _base_config(tmp_path)
    summary = _success_summary()
    summary.update(
        {
            "importance_generation_performed": False,
            "safe_stop": True,
            "num_samples": 0,
            "context_importance_shape_example": None,
            "importance_norm_min": None,
            "importance_norm_max": None,
        }
    )
    Path(config["output"]["importance_summary_json"]).write_text(json.dumps(summary), encoding="utf-8")
    result = evaluate_step25_importance(config_path)
    assert result["pass"] is False
    assert result["safe_stop"] is True
    assert result["safety_gate_pass"] is True
