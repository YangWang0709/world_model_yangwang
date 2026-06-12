import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_tfds_world_model_tiny_overfit import evaluate_step27_tiny_overfit


def _base_config(tmp_path):
    config = yaml.safe_load(
        Path("configs/bridgedata_v2_tfds_world_model_tiny_overfit_step27.yaml").read_text(encoding="utf-8")
    )
    for key in [
        "training_summary_json",
        "training_summary_md",
        "loss_curves_json",
        "policy_comparison_json",
        "final_eval_json",
        "eval_json",
        "eval_md",
    ]:
        suffix = ".md" if key.endswith("_md") else ".json"
        config["output"][key] = str(tmp_path / f"{key}{suffix}")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, path


def _success_summary():
    return {
        "stage": "bridgedata_v2_tfds_world_model_tiny_overfit_step27",
        "tiny_overfit_training_performed": True,
        "training_performed": True,
        "tiny_overfit_training_only": True,
        "safe_stop": False,
        "num_samples": 4,
        "policies_trained": ["current_only", "random_context_topk", "proxy_importance_topk", "full_context_reference"],
        "train_steps": 300,
        "optimizer": "adamw",
        "optimizer_step_performed": True,
        "optimizer_step_scope": "tiny_world_model_predictor_only",
        "all_losses_finite": True,
        "policies_with_loss_decrease": 4,
        "acceptance_pass": True,
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "current_importance_training_performed": False,
        "videomae_training_performed": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "world_model_large_training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "download_performed": False,
        "model_download_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "checkpoint_saved": False,
        "tiny_overfit_result_not_final_performance": True,
        "safety_gate_pass": True,
        "policy_metric_table": [],
    }


def test_success_report_passes_and_recommends_step28(tmp_path):
    config, config_path = _base_config(tmp_path)
    comparison = {
        "num_policies": 4,
        "policies_with_loss_decrease": 4,
        "all_losses_finite": True,
        "policy_metrics": [],
        "loss_quality_note": "tiny-overfit only; not final performance",
    }
    Path(config["output"]["training_summary_json"]).write_text(json.dumps(_success_summary()), encoding="utf-8")
    Path(config["output"]["loss_curves_json"]).write_text(json.dumps({"loss_curves": []}), encoding="utf-8")
    Path(config["output"]["policy_comparison_json"]).write_text(json.dumps(comparison), encoding="utf-8")
    Path(config["output"]["final_eval_json"]).write_text(json.dumps({"policy_comparison": comparison}), encoding="utf-8")
    result = evaluate_step27_tiny_overfit(config_path)
    assert result["pass"] is True
    assert result["safe_stop"] is False
    assert result["optimizer_step_scope"] == "tiny_world_model_predictor_only"
    assert result["loss_quality_note"] == "tiny-overfit only; not final performance"
    assert "Step28" in result["recommended_step28"]["name"] or "context predictor" in result["recommended_step28"]["name"]


def test_safe_stop_does_not_pretend_to_pass(tmp_path):
    config, config_path = _base_config(tmp_path)
    summary = _success_summary()
    summary.update(
        {
            "tiny_overfit_training_performed": False,
            "training_performed": False,
            "safe_stop": True,
            "num_samples": 0,
            "optimizer_step_performed": False,
            "all_losses_finite": False,
            "acceptance_pass": False,
        }
    )
    Path(config["output"]["training_summary_json"]).write_text(json.dumps(summary), encoding="utf-8")
    Path(config["output"]["policy_comparison_json"]).write_text(
        json.dumps({"policies_with_loss_decrease": 0, "all_losses_finite": False, "policy_metrics": []}),
        encoding="utf-8",
    )
    result = evaluate_step27_tiny_overfit(config_path)
    assert result["pass"] is False
    assert result["safe_stop"] is True
    assert result["safety_gate_pass"] is True
