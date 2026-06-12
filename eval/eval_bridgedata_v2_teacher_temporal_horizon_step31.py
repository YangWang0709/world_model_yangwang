"""Evaluate Step31 teacher/temporal/horizon diagnostic outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.bridgedata_v2_teacher_failure_analysis import build_teacher_failure_analysis

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_teacher_temporal_horizon_diagnosis_step31.yaml"
STEP_DOC = PROJECT_ROOT / "docs" / "STEP31_BRIDGEDATA_V2_TEACHER_TEMPORAL_HORIZON_DIAGNOSIS.md"
ARCH_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TEACHER_ARCHITECTURE_DIAGNOSIS_REPORT.md"
TEMPORAL_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TEMPORAL_HORIZON_DIAGNOSIS_REPORT.md"
FAILURE_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TEACHER_FAILURE_ANALYSIS.md"
DECISION_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_STEP31_DECISION.md"


def evaluate_step31_teacher_temporal_horizon(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved = Path(config_path).resolve()
    config = _load_yaml(config_path)
    output = config["output"]
    architecture = _read_json(output["teacher_architecture_ablation_json"])
    label_scores = _read_json(output["label_score_ablation_json"])
    temporal = _read_json(output["temporal_horizon_diagnosis_json"])
    dominance = _read_json(output["current_dominance_diagnosis_json"])
    step30b_eval_path = Path(config["input"]["step30b_run_dir"]) / "occlusion_teacher_eval.json"
    step30b_eval = _read_json(step30b_eval_path)
    failure = build_teacher_failure_analysis(architecture, label_scores, temporal, dominance, step30b_eval)
    decision = {
        "stage": config["stage"],
        "recommended_step32": failure["recommended_step32"],
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "do_not_train_selector_yet": True,
        "do_not_train_current_importance_yet": True,
        "never_claim_final_utility": True,
        "safety_gate_pass": True,
    }
    result = _result(config, architecture, label_scores, temporal, dominance, failure, decision)
    _write_json(Path(output["teacher_failure_analysis_json"]), failure)
    _write_json(Path(output["step31_decision_json"]), decision)
    _write_json(Path(output["eval_json"]), result)
    report = _format_eval_report(result)
    Path(output["eval_md"]).parent.mkdir(parents=True, exist_ok=True)
    Path(output["eval_md"]).write_text(report, encoding="utf-8")
    if resolved == DEFAULT_CONFIG.resolve():
        STEP_DOC.write_text(_format_step_doc(result), encoding="utf-8")
        ARCH_DOC.write_text(_format_architecture_doc(architecture, result), encoding="utf-8")
        TEMPORAL_DOC.write_text(_format_temporal_doc(temporal, dominance), encoding="utf-8")
        FAILURE_DOC.write_text(_format_failure_doc(failure), encoding="utf-8")
        DECISION_DOC.write_text(_format_decision_doc(decision, failure), encoding="utf-8")
    return result


def _result(
    config: dict[str, Any],
    architecture: dict[str, Any],
    label_scores: dict[str, Any],
    temporal: dict[str, Any],
    dominance: dict[str, Any],
    failure: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    forbidden = [
        architecture.get("token_extraction_performed"),
        architecture.get("proxy_importance_regeneration_performed"),
        architecture.get("teacher_label_regeneration_performed"),
        architecture.get("new_tfds_shard_downloaded"),
        architecture.get("model_download_performed"),
        architecture.get("videomae_training_performed"),
        architecture.get("selector_training_performed"),
        architecture.get("current_importance_training_performed"),
        temporal.get("token_extraction_performed"),
        temporal.get("proxy_importance_regeneration_performed"),
        temporal.get("teacher_label_regeneration_performed"),
        temporal.get("new_tfds_shard_downloaded"),
        temporal.get("model_download_performed"),
        temporal.get("videomae_training_performed"),
        temporal.get("selector_training_performed"),
        temporal.get("current_importance_training_performed"),
        decision.get("context_utility_claim_allowed"),
        decision.get("selector_training_allowed"),
        decision.get("current_importance_training_allowed"),
    ]
    safety_gate_pass = not any(bool(item) for item in forbidden)
    passed = (
        safety_gate_pass
        and bool(architecture.get("teacher_architecture_diagnosis_performed"))
        and bool(label_scores.get("label_score_ablation_performed"))
        and bool(temporal.get("temporal_horizon_diagnosis_performed"))
        and bool(dominance.get("current_dominance_diagnosis_performed"))
        and bool(failure.get("teacher_failure_analysis_performed"))
        and bool(decision.get("recommended_step32"))
    )
    return {
        "stage": config["stage"],
        "pass": bool(passed),
        "safe_stop": bool(architecture.get("safe_stop") or temporal.get("safe_stop")),
        "teacher_architecture_diagnosis_performed": bool(architecture.get("teacher_architecture_diagnosis_performed")),
        "label_score_ablation_performed": bool(label_scores.get("label_score_ablation_performed")),
        "temporal_horizon_diagnosis_performed": bool(temporal.get("temporal_horizon_diagnosis_performed")),
        "current_dominance_diagnosis_performed": bool(dominance.get("current_dominance_diagnosis_performed")),
        "teacher_failure_analysis_performed": bool(failure.get("teacher_failure_analysis_performed")),
        "diagnostic_predictor_training_performed": bool(
            architecture.get("diagnostic_predictor_training_performed")
            or temporal.get("diagnostic_predictor_training_performed")
        ),
        "teacher_topk_underperforms_proxy_confirmed": bool(label_scores.get("teacher_topk_underperforms_proxy_confirmed")),
        "best_diagnostic_score": label_scores.get("best_diagnostic_score"),
        "best_architecture_variant": architecture.get("best_architecture_variant"),
        "current_dominance_level": dominance.get("current_dominance_level"),
        "full_context_noise_confirmed": bool(failure.get("full_context_noise_confirmed")),
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "token_extraction_performed": False,
        "proxy_importance_regeneration_performed": False,
        "teacher_label_regeneration_performed": False,
        "new_tfds_shard_downloaded": False,
        "model_download_performed": False,
        "videomae_training_performed": False,
        "selector_training_performed": False,
        "current_importance_training_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "checkpoint_saved": False,
        "recommended_step32": decision.get("recommended_step32"),
        "teacher_architecture_ablation": architecture,
        "label_score_ablation": label_scores,
        "temporal_horizon_diagnosis": temporal,
        "current_dominance_diagnosis": dominance,
        "teacher_failure_analysis": failure,
        "step31_decision": decision,
        "safety_gate_pass": bool(safety_gate_pass),
    }


def _format_eval_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 Step31 Teacher Temporal Horizon Eval",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- safe_stop: `{str(result['safe_stop']).lower()}`",
            f"- diagnostic_predictor_training_performed: `{str(result['diagnostic_predictor_training_performed']).lower()}`",
            f"- best_architecture_variant: `{result['best_architecture_variant']}`",
            f"- best_diagnostic_score: `{result['best_diagnostic_score']}`",
            f"- teacher_topk_underperforms_proxy_confirmed: `{str(result['teacher_topk_underperforms_proxy_confirmed']).lower()}`",
            f"- current_dominance_level: `{result['current_dominance_level']}`",
            f"- full_context_noise_confirmed: `{str(result['full_context_noise_confirmed']).lower()}`",
            f"- recommended_step32: `{result['recommended_step32']}`",
            f"- context_utility_claim_allowed: `{str(result['context_utility_claim_allowed']).lower()}`",
            f"- selector_training_allowed: `{str(result['selector_training_allowed']).lower()}`",
            f"- current_importance_training_allowed: `{str(result['current_importance_training_allowed']).lower()}`",
            f"- safety_gate_pass: `{str(result['safety_gate_pass']).lower()}`",
            "",
        ]
    )


def _format_step_doc(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step31 BridgeData V2 Teacher Temporal Horizon Diagnosis",
            "",
            "Step31 diagnoses why Step30B trained-predictor occlusion labels did not beat Step30A proxy labels.",
            "",
            "- no selector training",
            "- no current importance training",
            "- no token extraction",
            "- no teacher label regeneration",
            "- no new data/model download",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- recommended_step32: `{result['recommended_step32']}`",
            "",
        ]
    )


def _format_architecture_doc(architecture: dict[str, Any], result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 Teacher Architecture Diagnosis Report",
            "",
            f"- best_architecture_variant: `{architecture.get('best_architecture_variant')}`",
            f"- proxy_prior_helped_attention: `{str(architecture.get('proxy_prior_helped_attention')).lower()}`",
            f"- diagnostic_predictor_training_performed: `{str(architecture.get('diagnostic_predictor_training_performed')).lower()}`",
            "",
            "```json",
            json.dumps(architecture.get("mean_val_loss_by_variant", {}), indent=2, sort_keys=True),
            "```",
            "",
            f"- selector_training_allowed: `{str(result['selector_training_allowed']).lower()}`",
            f"- current_importance_training_allowed: `{str(result['current_importance_training_allowed']).lower()}`",
            "",
        ]
    )


def _format_temporal_doc(temporal: dict[str, Any], dominance: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 Temporal Horizon Diagnosis Report",
            "",
            f"- best_target_variant: `{temporal.get('best_target_variant')}`",
            f"- delta_target_increases_context_gain: `{str(temporal.get('delta_target_increases_context_gain')).lower()}`",
            f"- current_dominance_level: `{dominance.get('current_dominance_level')}`",
            f"- full_context_noise_confirmed: `{str(temporal.get('full_context_noise_confirmed')).lower()}`",
            "",
            "```json",
            json.dumps(temporal.get("target_rows", []), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _format_failure_doc(failure: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 Teacher Failure Analysis",
            "",
            f"- teacher_topk_underperforms_proxy_confirmed: `{str(failure.get('teacher_topk_underperforms_proxy_confirmed')).lower()}`",
            f"- teacher_proxy_low_correlation_confirmed: `{str(failure.get('teacher_proxy_low_correlation_confirmed')).lower()}`",
            f"- h1_teacher_architecture_too_weak: `{str(failure.get('h1_teacher_architecture_too_weak')).lower()}`",
            f"- h2_occlusion_not_faithful: `{str(failure.get('h2_occlusion_not_faithful')).lower()}`",
            f"- h3_frame_repeat_temporal_limitation: `{str(failure.get('h3_frame_repeat_temporal_limitation')).lower()}`",
            f"- h4_short_horizon_current_dominance: `{str(failure.get('h4_short_horizon_current_dominance')).lower()}`",
            f"- h5_data_diversity_limited: `{str(failure.get('h5_data_diversity_limited')).lower()}`",
            "",
            "```json",
            json.dumps(failure.get("likely_failure_causes", []), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _format_decision_doc(decision: dict[str, Any], failure: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 Step31 Decision",
            "",
            "Recommended Step32:",
            str(decision.get("recommended_step32", {}).get("name")),
            "",
            "Do not claim final context utility.",
            "Do not train current importance yet.",
            "Do not train selector yet.",
            "",
            f"- condition: `{decision.get('recommended_step32', {}).get('condition')}`",
            f"- scope: `{decision.get('recommended_step32', {}).get('scope')}`",
            f"- best_architecture_variant: `{failure.get('best_architecture_variant')}`",
            f"- best_diagnostic_score: `{failure.get('best_diagnostic_score')}`",
            f"- current_dominance_level: `{failure.get('current_dominance_level')}`",
            f"- context_utility_claim_allowed: `{str(decision['context_utility_claim_allowed']).lower()}`",
            f"- selector_training_allowed: `{str(decision['selector_training_allowed']).lower()}`",
            f"- current_importance_training_allowed: `{str(decision['current_importance_training_allowed']).lower()}`",
            "",
        ]
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(evaluate_step31_teacher_temporal_horizon(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
