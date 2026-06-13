import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_factorized_selector_step37 import evaluate_step37_factorized_selector


def test_step37_eval_writes_reports_and_keeps_flags_false(tmp_path: Path):
    config = yaml.safe_load(
        Path("configs/bridgedata_v2_tfds_factorized_selector_step37.yaml").read_text(encoding="utf-8")
    )
    run = tmp_path / "run"
    config["output"]["run_dir"] = str(run)
    config["output"]["train_summary_json"] = str(run / "train_summary.json")
    config["output"]["loss_curves_json"] = str(run / "loss_curves.json")
    config["output"]["metrics_json"] = str(run / "factorized_selector_metrics.json")
    config["output"]["loss_ablation_json"] = str(run / "loss_ablation.json")
    config["output"]["gate_decision_json"] = str(run / "step37_gate_decision.json")
    config["output"]["eval_json"] = str(run / "step37_eval.json")
    config["output"]["eval_md"] = str(run / "step37_eval.md")
    run.mkdir()
    _write(
        config["output"]["train_summary_json"],
        {
            "stage": config["stage"],
            "safe_stop": False,
            "training_performed": True,
            "bounded_smoke_training_only": True,
            "factorized_selector_head_training_performed": True,
            "factorized_selector_with_proxy_temporal_prior_training_performed": True,
            "loss_ablation_performed": True,
            "optimizer_step_performed": True,
            "optimizer_scope": "proxy_factorized_selector_head_only",
            "videomae_training_performed": False,
            "videomae_frozen": True,
            "current_importance_training_performed": False,
            "final_selector_training_performed": False,
            "world_model_training_performed": False,
            "downstream_task_training_performed": False,
            "checkpoint_saved": False,
            "state_dict_saved": False,
            "data_token_shards_written": False,
            "data_importance_shards_written": False,
        },
    )
    _write(config["output"]["loss_curves_json"], {"stage": config["stage"], "curves": [], "num_curves": 1})
    metric = {
        "num_rows": 1,
        "factorized_selector_val_mse": 0.1,
        "random_token_baseline_mse": 0.2,
        "uniform_token_baseline_mse": 0.2,
        "temporal_broadcast_baseline_mse": 0.2,
        "current_only_patch_baseline_mse": 0.2,
        "step36_direct_patch_selector_mse_if_available": 0.2,
        "factorized_beats_random": True,
        "factorized_beats_uniform": True,
        "factorized_beats_temporal_broadcast": True,
        "factorized_beats_current_only": True,
        "factorized_beats_step36_direct_patch": True,
        "pearson": 0.5,
        "spearman": 0.5,
        "top64_overlap": 0.2,
        "top128_overlap": 0.2,
        "top256_overlap": 0.2,
        "top512_overlap": 0.2,
        "top256_precision": 0.2,
        "top256_recall": 0.2,
        "overfit_gap": 0.0,
    }
    _write(
        config["output"]["metrics_json"],
        {"stage": config["stage"], "within_shard": metric, "cross_shard": metric, "mixed_shard": metric, "safety_gate_pass": True},
    )
    _write(
        config["output"]["loss_ablation_json"],
        {
            "stage": config["stage"],
            "performed": True,
            "best_loss_variant": "mse_only",
            "best_factorized_selector_val_mse": 0.1,
            "rows": [],
        },
    )
    _write(
        config["output"]["gate_decision_json"],
        {
            "future_factorized_selector_gate_ready": True,
            "factorized_selector_beats_random_token_baseline": True,
            "factorized_selector_beats_uniform_token_baseline": True,
            "factorized_selector_beats_temporal_broadcast_baseline": True,
            "factorized_selector_beats_current_only_patch_baseline": True,
            "factorized_selector_beats_step36_direct_patch_selector": True,
            "factorized_selector_generalizes_cross_shard": True,
            "top256_overlap_mean": 0.2,
            "selector_training_allowed": False,
            "final_selector_training_allowed": False,
            "current_importance_training_allowed": False,
            "context_utility_claim_allowed": False,
            "recommended_step38": {"name": "next", "scope": "bounded"},
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = evaluate_step37_factorized_selector(config_path)

    assert summary["pass"] is True
    assert summary["selector_training_allowed"] is False
    assert summary["final_selector_training_allowed"] is False
    assert summary["current_importance_training_allowed"] is False
    assert summary["context_utility_claim_allowed"] is False
    assert Path(config["output"]["eval_json"]).exists()
    assert Path("docs/STEP37_FACTORIZED_SELECTOR_DIAGNOSIS.md").exists()


def _write(path: str, payload: dict):
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
