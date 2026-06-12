import json
from pathlib import Path

import yaml

from data.bridgedata_v2_tfds_world_model_smoke_metrics import LOSS_QUALITY_NOTE
from eval.eval_bridgedata_v2_tfds_world_model_smoke import evaluate_step26_world_model_smoke


def _base_config(tmp_path):
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_world_model_smoke_step26.yaml").read_text(encoding="utf-8"))
    for key in [
        "batch_summary_json",
        "policy_metrics_json",
        "forward_loss_json",
        "smoke_summary_json",
        "smoke_summary_md",
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
        "stage": "bridgedata_v2_tfds_world_model_smoke_step26",
        "world_model_smoke_performed": True,
        "safe_stop": False,
        "num_samples": 1,
        "policies_evaluated": ["current_only", "random_context_topk", "proxy_importance_topk", "full_context_reference"],
        "topk_values": [64, 128, 256],
        "all_losses_finite": True,
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "optimizer_step_performed": False,
        "training_performed": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "world_model_training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "download_performed": False,
        "model_download_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "random_init_result_not_scientific": True,
        "safety_gate_pass": True,
        "policy_metric_table": [],
    }


def test_success_report_passes_and_recommends_step27(tmp_path):
    config, config_path = _base_config(tmp_path)
    Path(config["output"]["batch_summary_json"]).write_text(json.dumps({"num_samples": 1}), encoding="utf-8")
    Path(config["output"]["policy_metrics_json"]).write_text(json.dumps({"policy_metrics": []}), encoding="utf-8")
    Path(config["output"]["forward_loss_json"]).write_text(
        json.dumps({"all_losses_finite": True, "loss_quality_note": LOSS_QUALITY_NOTE}),
        encoding="utf-8",
    )
    Path(config["output"]["smoke_summary_json"]).write_text(json.dumps(_success_summary()), encoding="utf-8")
    result = evaluate_step26_world_model_smoke(config_path)
    assert result["pass"] is True
    assert result["safe_stop"] is False
    assert result["loss_quality_note"] == LOSS_QUALITY_NOTE
    assert "Step27" in result["recommended_step27"]["name"]


def test_safe_stop_does_not_pretend_to_pass(tmp_path):
    config, config_path = _base_config(tmp_path)
    summary = _success_summary()
    summary.update(
        {
            "world_model_smoke_performed": False,
            "safe_stop": True,
            "num_samples": 0,
            "all_losses_finite": False,
        }
    )
    Path(config["output"]["smoke_summary_json"]).write_text(json.dumps(summary), encoding="utf-8")
    result = evaluate_step26_world_model_smoke(config_path)
    assert result["pass"] is False
    assert result["safe_stop"] is True
    assert result["safety_gate_pass"] is True
