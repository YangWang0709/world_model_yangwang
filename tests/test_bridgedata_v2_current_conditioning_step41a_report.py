import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_current_conditioning_step41a import evaluate_step41a_current_conditioning


def test_step41a_eval_writes_reports_and_keeps_forbidden_flags_false(tmp_path: Path):
    config = yaml.safe_load(
        Path("configs/bridgedata_v2_tfds_current_conditioning_diagnosis_step41a.yaml").read_text(encoding="utf-8")
    )
    run = tmp_path / "run"
    config["output"]["run_dir"] = str(run)
    config["output"]["train_summary_json"] = str(run / "train_summary.json")
    config["output"]["loss_curves_json"] = str(run / "loss_curves.json")
    config["output"]["current_conditioning_metrics_json"] = str(run / "metrics.json")
    config["output"]["variant_comparison_json"] = str(run / "comparison.json")
    config["output"]["gate_decision_json"] = str(run / "gate.json")
    config["output"]["eval_json"] = str(run / "eval.json")
    config["output"]["eval_md"] = str(run / "eval.md")
    run.mkdir()
    _write(
        config["output"]["train_summary_json"],
        {
            "stage": config["stage"],
            "safe_stop": False,
            "training_performed": True,
            "bounded_current_conditioning_smoke_training_performed": True,
            "optimizer_step_performed": True,
            "optimizer_scope": "current_conditioned_selector_head_only",
            "variants_run": ["no_current_context_only", "current_mean_summary"],
            "best_variant": "current_mean_summary",
            "label_variant": "global_spatial_prior_removed_residual",
            "videomae_loaded": False,
            "videomae_training_performed": False,
            "current_importance_training_performed": False,
            "final_selector_training_performed": False,
            "world_model_training_performed": False,
            "downstream_task_training_performed": False,
            "checkpoint_saved": False,
            "state_dict_saved": False,
            "data_token_shards_written": False,
            "data_importance_shards_written": False,
            "action_used_as_input": False,
            "language_used_as_input": False,
            "future_tokens_exposed_to_selector": False,
            "future_tokens_used_as_input": False,
        },
    )
    metric = _metric(0.1)
    comparison = {
        "stage": config["stage"],
        "best_variant": "current_mean_summary",
        "ranked_variants": [{"variant_name": "current_mean_summary", "selector_val_mse": 0.1}],
        "per_variant": {
            "no_current_context_only": {
                "overall": _metric(0.2),
                "within_shard": _metric(0.2),
                "cross_shard": _metric(0.2),
                "mixed_shard": _metric(0.2),
            },
            "current_mean_summary": {
                "overall": metric,
                "within_shard": metric,
                "cross_shard": metric,
                "mixed_shard": metric,
            },
        },
    }
    _write(config["output"]["loss_curves_json"], {"stage": config["stage"], "curves": [], "num_curves": 1})
    _write(
        config["output"]["current_conditioning_metrics_json"],
        {"stage": config["stage"], "best_variant": "current_mean_summary", "safety_gate_pass": True},
    )
    _write(config["output"]["variant_comparison_json"], comparison)
    _write(
        config["output"]["gate_decision_json"],
        {
            "stage": config["stage"],
            "current_conditioning_candidate_ready": True,
            "best_variant": "current_mean_summary",
            "best_variant_beats_no_current": True,
            "best_variant_beats_current_mean_summary": False,
            "best_variant_beats_random_token_baseline": True,
            "best_variant_beats_uniform_or_mean_baseline": True,
            "best_variant_beats_temporal_broadcast_baseline": True,
            "best_variant_beats_train_global_spatial_prior_baseline": True,
            "best_variant_generalizes_cross_shard": True,
            "top256_overlap_mean": 0.2,
            "overfit_gap_mean_abs": 0.01,
            "bounded_current_conditioning_smoke_performed": True,
            "downstream_selector_use_allowed": False,
            "final_selector_training_allowed": False,
            "current_importance_training_allowed": False,
            "context_utility_claim_allowed": False,
            "recommended_step42": {"name": "Step42 bounded smoke", "scope": "still no final selector"},
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = evaluate_step41a_current_conditioning(config_path)

    assert result["pass"] is True
    assert result["downstream_selector_use_allowed"] is False
    assert result["final_selector_training_allowed"] is False
    assert result["current_importance_training_allowed"] is False
    assert result["context_utility_claim_allowed"] is False
    assert Path(config["output"]["eval_json"]).exists()
    doc = Path("docs/STEP41A_CURRENT_CONDITIONING_DIAGNOSIS.md").read_text(encoding="utf-8")
    assert "not final selector training" in doc
    assert "keep VideoMAE frozen" in doc


def _metric(mse: float):
    return {
        "num_rows": 1,
        "selector_val_mse": mse,
        "selector_val_mae": mse,
        "random_token_baseline_mse": 0.9,
        "uniform_or_mean_baseline_mse": 0.8,
        "temporal_broadcast_baseline_mse": 0.7,
        "train_global_spatial_prior_baseline_mse": 0.6,
        "selector_beats_random_token_baseline": True,
        "selector_beats_uniform_or_mean_baseline": True,
        "selector_beats_temporal_broadcast_baseline": True,
        "selector_beats_train_global_spatial_prior_baseline": True,
        "beats_no_current_context_only": True,
        "beats_current_mean_summary": True,
        "pearson": 0.2,
        "spearman": 0.2,
        "top64_overlap": 0.2,
        "top128_overlap": 0.2,
        "top256_overlap": 0.2,
        "top512_overlap": 0.2,
        "top256_precision": 0.2,
        "top256_recall": 0.2,
        "train_mse": mse - 0.01,
        "overfit_gap": 0.01,
    }


def _write(path: str, payload: dict):
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
