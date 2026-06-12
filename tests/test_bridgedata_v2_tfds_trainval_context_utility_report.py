import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_tfds_trainval_context_utility import evaluate_step29_trainval_context_utility


def _base_config(tmp_path):
    config = yaml.safe_load(
        Path("configs/bridgedata_v2_tfds_trainval_context_utility_step29.yaml").read_text(encoding="utf-8")
    )
    out_dir = tmp_path / "outputs"
    for key in [
        "selected_windows_jsonl",
        "split_json",
        "token_summary_json",
        "importance_summary_json",
        "trainval_summary_json",
        "loss_curves_json",
        "policy_comparison_json",
        "context_utility_decision_json",
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
    Path(out["selected_windows_jsonl"]).write_text("\n".join("{}" for _ in range(32)) + "\n", encoding="utf-8")
    Path(out["split_json"]).write_text(
        json.dumps({"num_train_windows": 24, "num_val_windows": 8, "train_val_trajectory_disjoint": True}),
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
    Path(out["trainval_summary_json"]).write_text(
        json.dumps(
            {
                "safe_stop": False,
                "tiny_trainval_training_performed": True,
                "optimizer_step_performed": True,
                "optimizer_step_scope": "tiny_world_model_predictor_only",
                "all_val_losses_finite": True,
                "current_tokens_kept_full": True,
                "train_current_importance": False,
                "videomae_training_performed": False,
                "teacher_training_performed": False,
                "selector_training_performed": False,
                "current_importance_training_performed": False,
                "new_tfds_shard_downloaded": False,
                "model_download_performed": False,
                "data_token_shards_written": False,
                "data_importance_shards_written": False,
                "checkpoint_saved": False,
                "policies_trained": ["current_only", "random_context_topk", "proxy_importance_topk", "full_context_reference"],
            }
        ),
        encoding="utf-8",
    )
    Path(out["policy_comparison_json"]).write_text(
        json.dumps({"policy_metrics": [], "context_utility_sanity_signal": "inconclusive"}),
        encoding="utf-8",
    )
    Path(out["context_utility_decision_json"]).write_text(
        json.dumps(
            {
                "context_utility_sanity_signal": "inconclusive",
                "context_utility_claim_allowed": False,
                "recommended_step30": {
                    "name": "improve teacher label or increase window diversity before selector training",
                    "condition": "Step29 context utility sanity is inconclusive",
                    "scope": "no selector training yet",
                },
            }
        ),
        encoding="utf-8",
    )


def test_step29_success_report_passes_and_keeps_no_final_claim(tmp_path):
    config, config_path = _base_config(tmp_path)
    _write_success_outputs(config)
    result = evaluate_step29_trainval_context_utility(config_path)
    assert result["pass"] is True
    assert result["safe_stop"] is False
    assert result["context_utility_decision"]
    assert result["recommended_step30"]
    assert result["context_utility_claim_allowed"] is False


def test_step29_safe_stop_does_not_pretend_to_pass(tmp_path):
    config, config_path = _base_config(tmp_path)
    _write_success_outputs(config)
    summary = json.loads(Path(config["output"]["trainval_summary_json"]).read_text(encoding="utf-8"))
    summary["safe_stop"] = True
    summary["tiny_trainval_training_performed"] = False
    Path(config["output"]["trainval_summary_json"]).write_text(json.dumps(summary), encoding="utf-8")
    result = evaluate_step29_trainval_context_utility(config_path)
    assert result["pass"] is False
    assert result["safe_stop"] is True

