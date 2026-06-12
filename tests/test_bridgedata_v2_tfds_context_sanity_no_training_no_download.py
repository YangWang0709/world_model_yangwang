import json
from pathlib import Path

import yaml

from analysis.bridgedata_v2_tfds_context_sanity import run_context_sanity_analysis


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_config(tmp_path):
    config = yaml.safe_load(
        Path("configs/bridgedata_v2_tfds_context_sanity_step28.yaml").read_text(encoding="utf-8")
    )
    input_dir = tmp_path / "inputs"
    outputs = tmp_path / "runs" / "step28"
    paths = {
        "step24_token_summary_json": input_dir / "step24_summary.json",
        "step24_token_manifest_jsonl": input_dir / "step24_manifest.jsonl",
        "step25_importance_summary_json": input_dir / "step25_summary.json",
        "step25_importance_manifest_jsonl": input_dir / "step25_manifest.jsonl",
        "step26_smoke_summary_json": input_dir / "step26_summary.json",
        "step26_policy_metrics_json": input_dir / "step26_policy_metrics.json",
        "step27_training_summary_json": input_dir / "step27_summary.json",
        "step27_loss_curves_json": input_dir / "step27_loss_curves.json",
        "step27_policy_comparison_json": input_dir / "step27_policy_comparison.json",
        "step27_final_eval_json": input_dir / "step27_final_eval.json",
    }
    for key, path in paths.items():
        config["input"][key] = str(path)
    config["output"]["run_root"] = str(tmp_path / "runs")
    config["output"]["run_name"] = "step28"
    for key in [
        "sanity_summary_json",
        "context_utility_json",
        "memorization_risk_json",
        "importance_label_json",
        "teacher_plan_json",
        "next_step_decision_json",
        "eval_json",
        "eval_md",
    ]:
        suffix = ".md" if key.endswith("_md") else ".json"
        config["output"][key] = str(outputs / f"{key}{suffix}")
    _write_json(paths["step24_token_summary_json"], {"token_extraction_performed": True, "safe_stop": False})
    paths["step24_token_manifest_jsonl"].write_text("{}\n", encoding="utf-8")
    _write_json(
        paths["step25_importance_summary_json"],
        {
            "importance_generation_performed": True,
            "safe_stop": False,
            "num_samples": 4,
            "method": "proxy_token_mse_dryrun",
            "importance_raw_mean": 1.0e-6,
            "importance_raw_std": 2.0e-6,
            "importance_norm_min": 0.0,
            "importance_norm_max": 1.0,
            "label_quality_note": "proxy dry-run only; not final teacher label",
        },
    )
    paths["step25_importance_manifest_jsonl"].write_text("{}\n", encoding="utf-8")
    _write_json(paths["step26_smoke_summary_json"], {"world_model_smoke_performed": True, "safe_stop": False})
    _write_json(
        paths["step26_policy_metrics_json"],
        {
            "policy_metrics": [
                {"policy": "random_context_topk", "mean_selected_importance_mass": 0.04},
                {"policy": "proxy_importance_topk", "mean_selected_importance_mass": 0.24},
                {"policy": "full_context_reference", "mean_selected_importance_mass": 1.0},
            ]
        },
    )
    _write_json(
        paths["step27_training_summary_json"],
        {
            "tiny_overfit_training_performed": True,
            "safe_stop": False,
            "num_samples": 4,
            "loss_quality_note": "tiny-overfit only; not final performance",
        },
    )
    _write_json(paths["step27_loss_curves_json"], {"loss_curves": []})
    _write_json(
        paths["step27_policy_comparison_json"],
        {
            "policy_metrics": [
                {"policy": "current_only", "relative_loss_decrease": 0.99, "final_loss": 0.01, "best_loss": 0.01},
                {
                    "policy": "random_context_topk",
                    "relative_loss_decrease": 0.99,
                    "final_loss": 0.02,
                    "best_loss": 0.02,
                },
                {
                    "policy": "proxy_importance_topk",
                    "relative_loss_decrease": 0.98,
                    "final_loss": 0.03,
                    "best_loss": 0.011,
                },
                {
                    "policy": "full_context_reference",
                    "relative_loss_decrease": 0.99,
                    "final_loss": 0.01,
                    "best_loss": 0.01,
                },
            ]
        },
    )
    _write_json(paths["step27_final_eval_json"], {"pass": True})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, outputs


def test_analysis_run_writes_only_small_diagnostics_and_no_training_flags(tmp_path):
    config_path, outputs = _make_config(tmp_path)
    payload = run_context_sanity_analysis(config_path)
    summary = payload["summary"]
    assert summary["context_sanity_performed"] is True
    for key in [
        "download_performed",
        "model_download_performed",
        "training_performed",
        "optimizer_step_performed",
        "token_extraction_performed",
        "importance_generation_performed",
        "videomae_training_performed",
        "teacher_training_performed",
        "selector_training_performed",
        "current_importance_training_performed",
        "world_model_training_performed",
        "data_token_shards_written",
        "data_importance_shards_written",
        "checkpoint_saved",
    ]:
        assert summary[key] is False
    assert (outputs / "sanity_summary_json.json").exists()
    assert not (tmp_path / "data" / "token_shards").exists()
    assert not (tmp_path / "data" / "importance_shards").exists()

