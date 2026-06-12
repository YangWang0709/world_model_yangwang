import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_teacher_temporal_horizon_step31 import evaluate_step31_teacher_temporal_horizon


def _config(tmp_path):
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_teacher_temporal_horizon_diagnosis_step31.yaml").read_text(encoding="utf-8"))
    out_dir = tmp_path / "out"
    for key in config["output"]:
        suffix = ".md" if key.endswith("_md") else ".json"
        config["output"][key] = str(out_dir / f"{key}{suffix}")
    config["input"]["step30b_run_dir"] = str(out_dir)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, path


def _write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def _write_success(config):
    out = config["output"]
    _write_json(out["teacher_architecture_ablation_json"], {"teacher_architecture_diagnosis_performed": True, "diagnostic_predictor_training_performed": True, "best_architecture_variant": "current_plus_proxy_topk_context_predictor", "proxy_prior_helped_attention": False})
    _write_json(out["label_score_ablation_json"], {"label_score_ablation_performed": True, "best_diagnostic_score": "proxy_importance", "teacher_topk_underperforms_proxy_confirmed": True})
    _write_json(out["temporal_horizon_diagnosis_json"], {"temporal_horizon_diagnosis_performed": True, "full_context_noise_confirmed": True, "target_rows": []})
    _write_json(out["current_dominance_diagnosis_json"], {"current_dominance_diagnosis_performed": True, "current_dominance_level": "medium", "full_context_noise_penalty_present": True})
    _write_json(Path(config["input"]["step30b_run_dir"]) / "occlusion_teacher_eval.json", {"teacher_proxy_pearson_mean": 0.0})


def test_step31_success_report_passes_and_keeps_no_final_claim(tmp_path):
    config, path = _config(tmp_path)
    _write_success(config)
    result = evaluate_step31_teacher_temporal_horizon(path)
    assert result["pass"] is True
    assert result["recommended_step32"]
    assert result["context_utility_claim_allowed"] is False
    assert result["selector_training_allowed"] is False
    assert result["current_importance_training_allowed"] is False


def test_step31_safe_stop_does_not_pretend_to_pass(tmp_path):
    config, path = _config(tmp_path)
    _write_success(config)
    payload = json.loads(Path(config["output"]["teacher_architecture_ablation_json"]).read_text(encoding="utf-8"))
    payload["safe_stop"] = True
    payload["teacher_architecture_diagnosis_performed"] = False
    Path(config["output"]["teacher_architecture_ablation_json"]).write_text(json.dumps(payload), encoding="utf-8")
    result = evaluate_step31_teacher_temporal_horizon(path)
    assert result["pass"] is False
    assert result["safe_stop"] is True
