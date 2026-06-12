import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_tfds_true_temporal_step33a import evaluate_step33a_true_temporal


def test_step33a_eval_report_pass_and_safe_stop_not_confused(tmp_path: Path):
    config = _fake_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_fake_outputs(config, safe_stop=False)
    result = evaluate_step33a_true_temporal(config_path)
    assert result["pass"] is True
    assert result["safe_stop"] is False
    assert result["context_utility_claim_allowed"] is False
    assert result["selector_training_allowed"] is False
    assert result["current_importance_training_allowed"] is False
    assert result["recommended_step33b_or_step34"]["scope"] == "no selector/current-importance training yet"

    _write_fake_outputs(config, safe_stop=True)
    stopped = evaluate_step33a_true_temporal(config_path)
    assert stopped["pass"] is False
    assert stopped["safe_stop"] is True


def _fake_config(tmp_path: Path) -> dict:
    output = {
        "true_temporal_token_summary_json": str(tmp_path / "token.json"),
        "true_temporal_importance_summary_json": str(tmp_path / "importance.json"),
        "true_temporal_trainval_results_json": str(tmp_path / "trainval.json"),
        "representation_comparison_json": str(tmp_path / "comparison.json"),
        "step33a_decision_json": str(tmp_path / "decision.json"),
        "eval_json": str(tmp_path / "eval.json"),
        "eval_md": str(tmp_path / "eval.md"),
    }
    return {
        "stage": "bridgedata_v2_tfds_true_temporal_step33a",
        "comparison": {"primary_target": "future_delta_last_minus_current"},
        "output": output,
    }


def _write_fake_outputs(config: dict, *, safe_stop: bool) -> None:
    out = config["output"]
    _write(out["true_temporal_token_summary_json"], {
        "safe_stop": safe_stop,
        "limited_true_temporal_token_extraction_performed": True,
        "num_samples": 2,
        "raw_token_shapes": {"context_example": [1, 392, 768], "current_example": [1, 392, 768], "future_example": [1, 392, 768]},
        "summary_shapes": {"context_summary": [768], "current_summary": [768], "future_summary": [768]},
        "context_token_shape_example": [392, 768],
        "current_token_shape_example": [392, 768],
        "future_token_shape_example": [392, 768],
        "model_download_performed": False,
        "videomae_training_performed": False,
        "data_token_shards_written": False,
        "large_token_shards_generated": False,
    })
    _write(out["true_temporal_importance_summary_json"], {
        "safe_stop": False,
        "limited_true_temporal_proxy_importance_performed": True,
        "context_importance_shape_example": [392],
        "temporal_importance_shape_example": [2],
        "spatial_importance_shape_example": [196],
        "temporal_spatial_summary_available": True,
        "model_download_performed": False,
        "training_performed": False,
        "selector_training_performed": False,
        "train_current_importance": False,
        "current_importance_generated": False,
        "data_importance_shards_written": False,
    })
    _write(out["true_temporal_trainval_results_json"], {
        "safe_stop": False,
        "tiny_trainval_diagnosis_performed": True,
        "optimizer_step_performed": True,
        "new_tfds_shard_downloaded": False,
        "raw_zip_downloaded": False,
        "full_tfds_downloaded": False,
        "droid_downloaded": False,
        "model_download_performed": False,
        "videomae_training_performed": False,
        "selector_training_performed": False,
        "current_importance_training_performed": False,
        "action_conditioned_world_model_training_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "checkpoint_saved": False,
        "runs": [],
    })
    _write(out["representation_comparison_json"], {
        "frame_repeat_proxy_gain_mean": 0.2,
        "true_temporal_proxy_gain_mean": 0.3,
        "true_temporal_gain_over_frame_repeat": 0.1,
        "true_temporal_representation_helped": True,
        "true_temporal_proxy_beats_current_fraction": 1.0,
        "true_temporal_proxy_beats_random_fraction": 1.0,
        "full_context_noise_confirmed": True,
        "target_rows": [],
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
    })
    _write(out["step33a_decision_json"], {
        "recommended_step33b_or_step34": {
            "name": "scale true temporal gap0 delta target before selector training",
            "condition": "fake",
            "scope": "no selector/current-importance training yet",
        },
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
    })


def _write(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
