import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_tfds_window_diversity_step30a import evaluate_step30a_window_diversity


def _base_config(tmp_path):
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_window_diversity_step30a.yaml").read_text(encoding="utf-8"))
    out_dir = tmp_path / "outputs"
    for key in [
        "selected_windows_jsonl",
        "selection_summary_json",
        "multiseed_splits_json",
        "token_summary_json",
        "importance_summary_json",
        "trainval_runs_json",
        "stability_summary_json",
        "context_signal_decision_json",
        "eval_json",
        "eval_md",
    ]:
        suffix = ".jsonl" if key.endswith("_jsonl") else ".md" if key.endswith("_md") else ".json"
        config["output"][key] = str(out_dir / f"{key}{suffix}")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, path


def _write_success_outputs(config):
    out = config["output"]
    Path(out["selected_windows_jsonl"]).parent.mkdir(parents=True, exist_ok=True)
    Path(out["selected_windows_jsonl"]).write_text("\n".join("{}" for _ in range(64)) + "\n", encoding="utf-8")
    Path(out["selection_summary_json"]).write_text(
        json.dumps({"num_selected_windows": 64, "num_selected_trajectories": 10, "trajectory_window_counts": {}}),
        encoding="utf-8",
    )
    Path(out["multiseed_splits_json"]).write_text(
        json.dumps(
            {
                "num_splits": 3,
                "seeds": [42, 123, 999],
                "splits": [
                    {"split_seed": 42, "num_train_windows": 48, "num_val_windows": 16, "train_val_trajectory_disjoint": True},
                    {"split_seed": 123, "num_train_windows": 48, "num_val_windows": 16, "train_val_trajectory_disjoint": True},
                    {"split_seed": 999, "num_train_windows": 48, "num_val_windows": 16, "train_val_trajectory_disjoint": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    Path(out["token_summary_json"]).write_text(
        json.dumps(
            {
                "limited_token_extraction_performed": True,
                "context_token_shape_example": [16, 392, 768],
                "current_token_shape_example": [4, 392, 768],
                "future_token_shape_example": [4, 392, 768],
                "model_download_performed": False,
                "data_token_shards_written": False,
            }
        ),
        encoding="utf-8",
    )
    Path(out["importance_summary_json"]).write_text(
        json.dumps(
            {
                "limited_proxy_importance_generation_performed": True,
                "context_importance_shape_example": [16, 392],
                "temporal_importance_shape_example": [16],
                "spatial_importance_shape_example": [392],
                "data_importance_shards_written": False,
            }
        ),
        encoding="utf-8",
    )
    Path(out["trainval_runs_json"]).write_text(
        json.dumps(
            {
                "safe_stop": False,
                "tiny_trainval_training_performed": True,
                "optimizer_step_performed": True,
                "optimizer_step_scope": "tiny_world_model_predictor_only",
                "all_val_losses_finite": True,
                "policies_trained": ["current_only", "random_context_topk", "proxy_importance_topk", "full_context_reference"],
                "new_tfds_shard_downloaded": False,
                "model_download_performed": False,
                "videomae_training_performed": False,
                "teacher_training_performed": False,
                "selector_training_performed": False,
                "current_importance_training_performed": False,
                "data_token_shards_written": False,
                "data_importance_shards_written": False,
                "checkpoint_saved": False,
                "runs": [],
            }
        ),
        encoding="utf-8",
    )
    Path(out["stability_summary_json"]).write_text(
        json.dumps(
            {
                "proxy_beats_current_fraction": 0.67,
                "proxy_beats_random_fraction": 0.67,
                "full_beats_current_fraction": 0.33,
                "mean_proxy_improvement_over_current": 0.02,
                "mean_proxy_improvement_over_random": 0.02,
                "context_signal_stability": "weak_proxy_positive_full_context_noisy",
            }
        ),
        encoding="utf-8",
    )
    Path(out["context_signal_decision_json"]).write_text(
        json.dumps(
            {
                "context_signal_stability": "weak_proxy_positive_full_context_noisy",
                "context_utility_claim_allowed": False,
                "selector_training_allowed": False,
                "current_importance_training_allowed": False,
                "recommended_step30b": {
                    "name": "trained-predictor occlusion teacher or add more data diversity",
                    "condition": "proxy top-k is stable but full context is noisy",
                    "scope": "no selector/current-importance training yet",
                },
            }
        ),
        encoding="utf-8",
    )


def test_step30a_success_report_passes_and_keeps_no_final_claim(tmp_path):
    config, config_path = _base_config(tmp_path)
    _write_success_outputs(config)
    result = evaluate_step30a_window_diversity(config_path)
    assert result["pass"] is True
    assert result["safe_stop"] is False
    assert result["context_signal_decision"]
    assert result["recommended_step30b"]
    assert result["context_utility_claim_allowed"] is False
    assert result["selector_training_allowed"] is False
    assert result["current_importance_training_allowed"] is False


def test_step30a_safe_stop_does_not_pretend_to_pass(tmp_path):
    config, config_path = _base_config(tmp_path)
    _write_success_outputs(config)
    summary = json.loads(Path(config["trainval_runs_json"] if "trainval_runs_json" in config else config["output"]["trainval_runs_json"]).read_text(encoding="utf-8"))
    summary["safe_stop"] = True
    summary["tiny_trainval_training_performed"] = False
    Path(config["output"]["trainval_runs_json"]).write_text(json.dumps(summary), encoding="utf-8")
    result = evaluate_step30a_window_diversity(config_path)
    assert result["pass"] is False
    assert result["safe_stop"] is True
