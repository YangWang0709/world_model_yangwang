import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_proxy_temporal_selector_step35 import evaluate_step35_proxy_temporal_selector


def test_step35_eval_writes_reports_and_keeps_flags_false(tmp_path: Path):
    config = yaml.safe_load(
        Path("configs/bridgedata_v2_tfds_proxy_temporal_selector_train_step35.yaml").read_text(encoding="utf-8")
    )
    run = tmp_path / "run"
    config["output"]["run_dir"] = str(run)
    config["output"]["train_summary_json"] = str(run / "train_summary.json")
    config["output"]["loss_curves_json"] = str(run / "loss_curves.json")
    config["output"]["metrics_json"] = str(run / "selector_metrics.json")
    config["output"]["gate_decision_json"] = str(run / "step35_gate_decision.json")
    config["output"]["eval_json"] = str(run / "step35_eval.json")
    config["output"]["eval_md"] = str(run / "step35_eval.md")
    run.mkdir()
    _write(
        config["output"]["train_summary_json"],
        {
            "stage": config["stage"],
            "safe_stop": False,
            "training_performed": True,
            "bounded_smoke_training_only": True,
            "temporal_selector_head_training_performed": True,
            "optimizer_step_performed": True,
            "optimizer_scope": "proxy_temporal_selector_head_only",
            "videomae_training_performed": False,
            "videomae_frozen": True,
            "current_importance_training_performed": False,
            "final_selector_training_performed": False,
            "patch_level_selector_training_performed": False,
            "checkpoint_saved": False,
            "data_token_shards_written": False,
            "data_importance_shards_written": False,
        },
    )
    _write(config["output"]["loss_curves_json"], {"stage": config["stage"], "curves": [], "num_curves": 1})
    metric = {
        "num_rows": 1,
        "selector_val_mse": 0.1,
        "random_baseline_mse": 0.2,
        "current_only_baseline_mse": 0.2,
        "uniform_baseline_mse": 0.2,
        "selector_beats_random_baseline": True,
        "selector_beats_current_only_baseline": True,
        "selector_beats_uniform_baseline": True,
        "selector_spearman": 0.5,
        "selector_pearson": 0.5,
        "top1_frame_hit": 1.0,
        "top2_frame_overlap": 1.0,
        "top4_frame_overlap": 1.0,
        "overfit_gap": 0.0,
    }
    _write(
        config["output"]["metrics_json"],
        {"stage": config["stage"], "within_shard": metric, "cross_shard": metric, "mixed_shard": metric, "safety_gate_pass": True},
    )
    _write(
        config["output"]["gate_decision_json"],
        {
            "future_selector_training_gate_ready": True,
            "selector_beats_random_baseline": True,
            "selector_beats_current_only_baseline": True,
            "selector_generalizes_cross_shard": True,
            "selector_training_allowed": False,
            "final_selector_training_allowed": False,
            "current_importance_training_allowed": False,
            "context_utility_claim_allowed": False,
            "recommended_step36": {"name": "next", "scope": "bounded"},
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = evaluate_step35_proxy_temporal_selector(config_path)

    assert summary["pass"] is True
    assert summary["selector_training_allowed"] is False
    assert summary["final_selector_training_allowed"] is False
    assert summary["current_importance_training_allowed"] is False
    assert summary["context_utility_claim_allowed"] is False
    assert Path(config["output"]["eval_json"]).exists()
    assert Path("docs/STEP35_PROXY_TEMPORAL_SELECTOR_TRAINING_SMOKE.md").exists()


def _write(path: str, payload: dict):
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
