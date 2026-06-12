import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_tfds_context_sanity import evaluate_step28_context_sanity


def _base_config(tmp_path):
    config = yaml.safe_load(
        Path("configs/bridgedata_v2_tfds_context_sanity_step28.yaml").read_text(encoding="utf-8")
    )
    out_dir = tmp_path / "outputs"
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
        config["output"][key] = str(out_dir / f"{key}{suffix}")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, path


def _success_summary():
    return {
        "stage": "bridgedata_v2_tfds_context_sanity_step28",
        "context_sanity_performed": True,
        "safe_stop": False,
        "step24_pass": True,
        "step25_pass": True,
        "step26_pass": True,
        "step27_pass": True,
        "num_samples": 4,
        "current_only_can_overfit": True,
        "current_only_relative_loss_decrease": 0.9997,
        "memorization_risk": "high",
        "proxy_importance_mass_advantage_over_random": 6.6,
        "context_utility_claim_allowed": False,
        "importance_label_is_proxy_only": True,
        "train_current_importance_now": False,
        "download_performed": False,
        "model_download_performed": False,
        "training_performed": False,
        "optimizer_step_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "videomae_training_performed": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "current_importance_training_performed": False,
        "world_model_training_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "checkpoint_saved": False,
        "safety_gate_pass": True,
    }


def _write_outputs(config, summary):
    outputs = config["output"]
    Path(outputs["sanity_summary_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(outputs["sanity_summary_json"]).write_text(json.dumps(summary), encoding="utf-8")
    Path(outputs["context_utility_json"]).write_text(
        json.dumps({"context_utility_claim_allowed": False}), encoding="utf-8"
    )
    Path(outputs["memorization_risk_json"]).write_text(
        json.dumps({"memorization_risk": summary["memorization_risk"], "requires_train_val_split": True}),
        encoding="utf-8",
    )
    Path(outputs["importance_label_json"]).write_text(
        json.dumps(
            {
                "label_is_proxy": summary["importance_label_is_proxy_only"],
                "label_quality_note": "proxy dry-run only; not final teacher label",
            }
        ),
        encoding="utf-8",
    )
    Path(outputs["teacher_plan_json"]).write_text(
        json.dumps({"candidate_plans": [{"name": "trained predictor occlusion teacher"}]}), encoding="utf-8"
    )
    Path(outputs["next_step_decision_json"]).write_text(
        json.dumps(
            {
                "recommended_step29": {
                    "name": "BridgeData V2 TFDS 32/64-window train-val context utility sanity",
                    "condition": "Step28 shows 4-sample overfit is memorization-prone",
                    "scope": "use existing shard first; allow limited token extraction expansion, not full training",
                },
                "alternative_step29": {
                    "name": "trained-predictor occlusion teacher dry-run",
                    "condition": "if user chooses teacher-first path",
                    "scope": "no selector training yet",
                },
            }
        ),
        encoding="utf-8",
    )


def test_success_report_passes_and_keeps_context_claim_off(tmp_path):
    config, config_path = _base_config(tmp_path)
    _write_outputs(config, _success_summary())
    result = evaluate_step28_context_sanity(config_path)
    assert result["pass"] is True
    assert result["safe_stop"] is False
    assert result["recommended_step29"]["name"]
    assert result["context_utility_claim_allowed"] is False
    assert result["label_quality_note"] == "proxy dry-run only; not final teacher label"
    assert result["train_current_importance_now"] is False


def test_safe_stop_does_not_pretend_to_pass(tmp_path):
    config, config_path = _base_config(tmp_path)
    summary = _success_summary()
    summary.update({"context_sanity_performed": False, "safe_stop": True, "step24_pass": False})
    _write_outputs(config, summary)
    result = evaluate_step28_context_sanity(config_path)
    assert result["pass"] is False
    assert result["safe_stop"] is True
    assert result["safety_gate_pass"] is True

