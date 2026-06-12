import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_tfds_longer_horizon_step32 import evaluate_step32_longer_horizon


def test_step32_eval_report_pass_and_safe_stop_not_confused(tmp_path: Path):
    config = _fake_config(tmp_path)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_fake_outputs(config, safe_stop=False)
    result = evaluate_step32_longer_horizon(tmp_path / "config.yaml")
    assert result["pass"] is True
    assert result["safe_stop"] is False
    assert result["context_utility_claim_allowed"] is False
    assert result["selector_training_allowed"] is False
    assert result["current_importance_training_allowed"] is False
    assert result["recommended_step33"]["scope"] == "no selector/current-importance training yet"

    _write_fake_outputs(config, safe_stop=True)
    stopped = evaluate_step32_longer_horizon(tmp_path / "config.yaml")
    assert stopped["pass"] is False
    assert stopped["safe_stop"] is True


def _fake_config(tmp_path: Path) -> dict:
    output = {
        "horizon_window_summary_json": str(tmp_path / "window.json"),
        "horizon_split_json": str(tmp_path / "splits.json"),
        "clip_export_summary_json": str(tmp_path / "clip.json"),
        "token_summary_json": str(tmp_path / "token.json"),
        "importance_summary_json": str(tmp_path / "importance.json"),
        "trainval_results_json": str(tmp_path / "trainval.json"),
        "horizon_target_summary_json": str(tmp_path / "target.json"),
        "step32_decision_json": str(tmp_path / "decision.json"),
        "eval_json": str(tmp_path / "eval.json"),
        "eval_md": str(tmp_path / "eval.md"),
    }
    return {
        "stage": "bridgedata_v2_tfds_longer_horizon_step32",
        "window": {"required_horizon_gaps": [0, 4, 8], "optional_horizon_gaps": [12]},
        "output": output,
    }


def _write_fake_outputs(config: dict, *, safe_stop: bool) -> None:
    out = config["output"]
    _write(out["horizon_window_summary_json"], {
        "safe_stop": safe_stop,
        "longer_horizon_window_builder_performed": True,
        "horizons_completed": [0, 4, 8],
        "horizons": {
            "gap0": {"selected_windows": 64, "num_trajectories": 10},
            "gap4": {"selected_windows": 64, "num_trajectories": 10},
            "gap8": {"selected_windows": 64, "num_trajectories": 8},
        },
        "new_tfds_shard_downloaded": False,
        "raw_zip_downloaded": False,
    })
    _write(out["horizon_split_json"], {"seeds": [42, 123, 999], "splits": []})
    _write(out["clip_export_summary_json"], {"clip_export_performed": True})
    _write(out["token_summary_json"], {
        "safe_stop": False,
        "limited_clip_export_performed": True,
        "limited_token_extraction_performed": True,
        "token_shapes_ok": True,
        "context_token_shape_example": [16, 392, 768],
        "current_token_shape_example": [4, 392, 768],
        "future_token_shape_example": [4, 392, 768],
        "new_tfds_shard_downloaded": False,
        "model_download_performed": False,
        "data_token_shards_written": False,
    })
    _write(out["importance_summary_json"], {
        "safe_stop": False,
        "limited_proxy_importance_generation_performed": True,
        "context_importance_shape_example": [16, 392],
        "temporal_importance_shape_example": [16],
        "spatial_importance_shape_example": [392],
        "data_importance_shards_written": False,
    })
    _write(out["trainval_results_json"], {
        "safe_stop": False,
        "tiny_trainval_diagnosis_performed": True,
        "optimizer_step_performed": True,
        "new_tfds_shard_downloaded": False,
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
    _write(out["horizon_target_summary_json"], {
        "best_horizon_gap": 8,
        "best_target_variant": "future_delta_last_minus_current",
        "delta_target_amplifies_context_gain": True,
        "delta_target_consistently_amplifies_context_gain": True,
        "proxy_beats_current_fraction": 1.0,
        "proxy_beats_random_fraction": 1.0,
        "full_context_noise_confirmed": True,
        "horizon_target_rows": [],
        "recommended_step33": {
            "name": "true temporal token extraction ablation for the best longer-horizon delta target",
            "condition": "fake",
            "scope": "no selector/current-importance training yet",
        },
    })
    _write(out["step32_decision_json"], {
        "recommended_step33": {
            "name": "true temporal token extraction ablation for the best longer-horizon delta target",
            "condition": "fake",
            "scope": "no selector/current-importance training yet",
        },
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
    })


def _write(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload), encoding="utf-8")

