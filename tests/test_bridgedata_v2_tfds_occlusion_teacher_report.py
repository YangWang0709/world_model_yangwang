import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_tfds_occlusion_teacher_step30b import evaluate_step30b_occlusion_teacher


def _base_config(tmp_path):
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_occlusion_teacher_step30b.yaml").read_text(encoding="utf-8"))
    out_dir = tmp_path / "outputs"
    for key in [
        "teacher_train_summary_json",
        "teacher_loss_curves_json",
        "teacher_importance_summary_json",
        "teacher_proxy_comparison_json",
        "teacher_topk_utility_json",
        "teacher_decision_json",
        "eval_json",
        "eval_md",
    ]:
        suffix = ".md" if key.endswith("_md") else ".json"
        config["output"][key] = str(out_dir / f"{key}{suffix}")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, path


def _write_success_outputs(config):
    out = config["output"]
    Path(out["teacher_train_summary_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(out["teacher_train_summary_json"]).write_text(
        json.dumps(
            {
                "safe_stop": False,
                "trained_predictor_teacher_training_performed": True,
                "teacher_training_scope": "small_predictor_teacher_only",
                "split_seeds_trained": [42, 123, 999],
                "teacher_seed_summaries": [],
                "teacher_val_loss_finite": True,
                "token_extraction_performed": False,
                "proxy_importance_regeneration_performed": False,
                "new_tfds_shard_downloaded": False,
                "model_download_performed": False,
                "videomae_training_performed": False,
                "context_teacher_large_training_performed": False,
                "selector_training_performed": False,
                "current_importance_training_performed": False,
                "checkpoint_saved": False,
            }
        ),
        encoding="utf-8",
    )
    Path(out["teacher_importance_summary_json"]).write_text(
        json.dumps(
            {
                "safe_stop": False,
                "teacher_occlusion_importance_generated": True,
                "num_samples": 64,
                "context_importance_shape_example": [16, 392],
                "fallback_used_count": 0,
                "token_extraction_performed": False,
                "proxy_importance_regeneration_performed": False,
                "data_importance_shards_written": False,
            }
        ),
        encoding="utf-8",
    )
    Path(out["teacher_proxy_comparison_json"]).write_text(
        json.dumps(
            {
                "teacher_proxy_comparison_performed": True,
                "teacher_label_nontrivial": True,
                "teacher_proxy_pearson_mean": 0.1,
                "teacher_proxy_spearman_mean": 0.1,
                "topk_overlap": {"256": 0.2},
                "safety_gate_pass": True,
            }
        ),
        encoding="utf-8",
    )
    Path(out["teacher_topk_utility_json"]).write_text(
        json.dumps(
            {
                "teacher_topk_utility_performed": True,
                "teacher_beats_current_fraction": 1.0,
                "teacher_beats_random_fraction": 1.0,
                "teacher_beats_proxy_fraction": 0.5,
                "teacher_mean_improvement_over_proxy": 0.0,
                "per_seed_val_table": [],
                "selector_training_performed": False,
                "current_importance_training_performed": False,
                "checkpoint_saved": False,
            }
        ),
        encoding="utf-8",
    )
    Path(out["teacher_decision_json"]).write_text(
        json.dumps(
            {
                "recommended_step31": {
                    "name": "use teacher label as stronger target and scale windows/shards before selector training",
                    "condition": "teacher_topK is close to proxy_topK",
                    "scope": "no selector/current-importance training yet",
                },
                "context_utility_claim_allowed": False,
                "selector_training_allowed": False,
                "current_importance_training_allowed": False,
            }
        ),
        encoding="utf-8",
    )


def test_step30b_success_report_passes_and_keeps_no_final_claim(tmp_path):
    config, config_path = _base_config(tmp_path)
    _write_success_outputs(config)
    result = evaluate_step30b_occlusion_teacher(config_path)
    assert result["pass"] is True
    assert result["safe_stop"] is False
    assert result["recommended_step31"]
    assert result["context_utility_claim_allowed"] is False
    assert result["selector_training_allowed"] is False
    assert result["current_importance_training_allowed"] is False


def test_step30b_safe_stop_does_not_pretend_to_pass(tmp_path):
    config, config_path = _base_config(tmp_path)
    _write_success_outputs(config)
    summary = json.loads(Path(config["output"]["teacher_train_summary_json"]).read_text(encoding="utf-8"))
    summary["safe_stop"] = True
    summary["trained_predictor_teacher_training_performed"] = False
    Path(config["output"]["teacher_train_summary_json"]).write_text(json.dumps(summary), encoding="utf-8")
    result = evaluate_step30b_occlusion_teacher(config_path)
    assert result["pass"] is False
    assert result["safe_stop"] is True
