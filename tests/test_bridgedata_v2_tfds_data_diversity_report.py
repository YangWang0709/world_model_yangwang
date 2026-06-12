import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_tfds_data_diversity_step33b import evaluate_step33b_data_diversity


def test_step33b_eval_report_success_and_safe_stop_not_confused(tmp_path: Path):
    config = _fake_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_fake_outputs(config, safe_stop=False)
    result = evaluate_step33b_data_diversity(config_path)
    assert result["pass"] is True
    assert result["safe_stop"] is False
    assert result["context_utility_claim_allowed"] is False
    assert result["selector_training_allowed"] is False
    assert result["current_importance_training_allowed"] is False
    assert result["recommended_step34"]["scope"].startswith("no selector")

    _write_fake_outputs(config, safe_stop=True)
    stopped = evaluate_step33b_data_diversity(config_path)
    assert stopped["pass"] is False
    assert stopped["safe_stop"] is True


def _fake_config(tmp_path: Path) -> dict:
    out = {
        "shard_acquisition_summary_json": str(tmp_path / "acq.json"),
        "shard_inventory_json": str(tmp_path / "inventory.json"),
        "shard1_window_summary_json": str(tmp_path / "windows.json"),
        "token_summary_json": str(tmp_path / "tokens.json"),
        "importance_summary_json": str(tmp_path / "importance.json"),
        "within_shard_results_json": str(tmp_path / "within.json"),
        "cross_shard_results_json": str(tmp_path / "cross.json"),
        "mixed_shard_results_json": str(tmp_path / "mixed.json"),
        "dataset_bias_summary_json": str(tmp_path / "bias.json"),
        "data_diversity_decision_json": str(tmp_path / "decision.json"),
        "eval_json": str(tmp_path / "eval.json"),
        "eval_md": str(tmp_path / "eval.md"),
    }
    return {
        "stage": "bridgedata_v2_tfds_data_diversity_step33b",
        "window_selection": {"min_windows_per_shard": 32},
        "output": out,
    }


def _write_fake_outputs(config: dict, *, safe_stop: bool) -> None:
    out = config["output"]
    _write(out["shard_acquisition_summary_json"], {
        "safe_stop": safe_stop,
        "second_shard_available": not safe_stop,
        "downloaded_new_shard_count": 1,
        "selected_shard_index": 1,
        "selected_shard_filename": "bridge_dataset-train.tfrecord-00001-of-01024",
        "raw_zip_downloaded": False,
        "full_tfds_downloaded": False,
        "safety_gate_pass": True,
    })
    _write(out["shard_inventory_json"], {"local_shard_indices": [0, 1]})
    _write(out["shard1_window_summary_json"], {"safe_stop": safe_stop, "selected_windows": 64, "num_trajectories": 4})
    _write(out["token_summary_json"], {
        "safe_stop": safe_stop,
        "limited_token_extraction_performed": True,
        "token_shapes_ok": True,
        "context_token_shape_example": [16, 392, 768],
        "current_token_shape_example": [4, 392, 768],
        "future_token_shape_example": [4, 392, 768],
        "model_download_performed": False,
        "training_performed": False,
    })
    _write(out["importance_summary_json"], {
        "limited_proxy_importance_generation_performed": True,
        "context_importance_shape_example": [16, 392],
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "current_importance_generated": False,
    })
    row_summary = {"num_rows": 3, "proxy_beats_current_fraction": 1.0, "proxy_beats_random_fraction": 1.0, "proxy_gain_over_current_mean": 0.2, "full_context_noise_confirmed": True}
    _write(out["within_shard_results_json"], row_summary)
    _write(out["cross_shard_results_json"], row_summary)
    _write(out["mixed_shard_results_json"], row_summary)
    _write(out["dataset_bias_summary_json"], {"dataset_bias_detected": False, "distribution_shift_flags": {}})
    _write(out["data_diversity_decision_json"], {
        "cross_shard_eval_performed": True,
        "cross_shard_proxy_signal_stable": True,
        "mixed_shard_eval_performed": True,
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "recommended_step34": {"name": "prepare proxy-supervised selector training plan after cross-shard validation", "condition": "fake", "scope": "no selector/current-importance training yet unless explicitly approved after review"},
    })


def _write(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
