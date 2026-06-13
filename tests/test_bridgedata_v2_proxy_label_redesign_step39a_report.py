import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_proxy_label_redesign_step39a import evaluate_step39a_proxy_label_redesign


def test_step39a_eval_writes_reports_and_keeps_flags_false(tmp_path: Path):
    config = yaml.safe_load(
        Path("configs/bridgedata_v2_tfds_proxy_label_redesign_step39a.yaml").read_text(encoding="utf-8")
    )
    run = tmp_path / "run"
    config["output"]["run_dir"] = str(run)
    config["output"]["label_redesign_summary_json"] = str(run / "summary.json")
    config["output"]["label_variant_metrics_json"] = str(run / "metrics.json")
    config["output"]["label_variant_comparison_json"] = str(run / "comparison.json")
    config["output"]["gate_decision_json"] = str(run / "gate.json")
    config["output"]["eval_json"] = str(run / "eval.json")
    config["output"]["eval_md"] = str(run / "eval.md")
    run.mkdir()
    best = {
        "variant_name": "temporal_broadcast",
        "diagnosis_score": 0.7,
        "score_is_engineering_heuristic": True,
    }
    _write(
        config["output"]["label_redesign_summary_json"],
        {
            "stage": config["stage"],
            "safe_stop": False,
            "label_redesign_performed": True,
            "num_samples": 2,
            "num_variants": 1,
            "variant_names": ["temporal_broadcast"],
            "global_spatial_prior_stats": {"shape": [392]},
            "best_variant": best,
            "training_performed": False,
            "optimizer_step_performed": False,
            "checkpoint_saved": False,
            "state_dict_saved": False,
            "data_token_shards_written": False,
            "data_importance_shards_written": False,
            "new_dataset_download_performed": False,
            "new_model_download_performed": False,
            "action_used_as_input": False,
            "language_used_as_input": False,
            "future_tokens_used_as_input": False,
        },
    )
    _write(
        config["output"]["label_variant_metrics_json"],
        {
            "stage": config["stage"],
            "num_rows": 2,
            "variant_names": ["temporal_broadcast"],
            "aggregates": {},
            "training_performed": False,
            "optimizer_step_performed": False,
            "checkpoint_saved": False,
            "state_dict_saved": False,
        },
    )
    _write(
        config["output"]["label_variant_comparison_json"],
        {
            "stage": config["stage"],
            "ranking": {"ranked_variants": [best], "best_variant": best},
            "best_variant": best,
            "redesigned_tensor_artifacts_saved": False,
        },
    )
    _write(
        config["output"]["gate_decision_json"],
        {
            "stage": config["stage"],
            "redesigned_label_candidate_ready": True,
            "recommended_step40_label_variant": "temporal_broadcast",
            "recommended_step40_scope": "bounded selector smoke only",
            "recommended_step40_reason": "fake best variant",
            "selector_training_allowed": False,
            "final_selector_training_allowed": False,
            "current_importance_training_allowed": False,
            "context_utility_claim_allowed": False,
            "training_performed": False,
            "optimizer_step_performed": False,
            "checkpoint_saved": False,
            "state_dict_saved": False,
            "data_token_shards_written": False,
            "data_importance_shards_written": False,
            "new_dataset_download_performed": False,
            "new_model_download_performed": False,
            "action_used_as_input": False,
            "language_used_as_input": False,
            "future_tokens_used_as_input": False,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = evaluate_step39a_proxy_label_redesign(config_path)

    assert result["pass"] is True
    assert result["selector_training_allowed"] is False
    assert result["final_selector_training_allowed"] is False
    assert result["current_importance_training_allowed"] is False
    assert result["context_utility_claim_allowed"] is False
    assert Path(config["output"]["eval_json"]).exists()
    doc = Path("docs/STEP39A_PROXY_LABEL_REDESIGN.md").read_text(encoding="utf-8")
    assert "not selector training" in doc
    assert "downloaded new data/model: `false`" in doc


def _write(path: str, payload: dict):
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
