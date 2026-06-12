"""Evaluate Step30B trained-predictor occlusion teacher outputs."""

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

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_occlusion_teacher_step30b.yaml"
STEP_DOC = PROJECT_ROOT / "docs" / "STEP30B_BRIDGEDATA_V2_TFDS_OCCLUSION_TEACHER.md"
REPORT_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TFDS_OCCLUSION_TEACHER_REPORT.md"
COMPARISON_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TFDS_TEACHER_PROXY_COMPARISON.md"
DECISION_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TFDS_OCCLUSION_TEACHER_DECISION.md"


def evaluate_step30b_occlusion_teacher(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved_config = Path(config_path).resolve()
    config = _load_yaml(config_path)
    output = config["output"]
    train_summary = _read_json_or_empty(output["teacher_train_summary_json"])
    label_summary = _read_json_or_empty(output["teacher_importance_summary_json"])
    comparison = _read_json_or_empty(output["teacher_proxy_comparison_json"])
    topk = _read_json_or_empty(output["teacher_topk_utility_json"])
    decision = _read_json_or_empty(output["teacher_decision_json"])
    safety_gate_pass = _safety_gate_pass(train_summary, label_summary, comparison, topk, decision)
    safe_stop = bool(train_summary.get("safe_stop") or label_summary.get("safe_stop") or topk.get("safe_stop"))
    result = {
        "stage": config["stage"],
        "pass": _pass(train_summary, label_summary, comparison, topk, decision, safety_gate_pass),
        "safe_stop": safe_stop,
        "trained_predictor_teacher_training_performed": bool(train_summary.get("trained_predictor_teacher_training_performed")),
        "teacher_training_scope": train_summary.get("teacher_training_scope"),
        "split_seeds_trained": train_summary.get("split_seeds_trained") or [],
        "teacher_seed_summaries": train_summary.get("teacher_seed_summaries") or [],
        "teacher_val_loss_finite": bool(train_summary.get("teacher_val_loss_finite")),
        "teacher_occlusion_importance_generated": bool(label_summary.get("teacher_occlusion_importance_generated")),
        "num_samples": int(label_summary.get("num_samples") or 0),
        "teacher_label_shape": label_summary.get("context_importance_shape_example"),
        "fallback_used_count": int(label_summary.get("fallback_used_count") or 0),
        "teacher_label_nontrivial": bool(comparison.get("teacher_label_nontrivial")),
        "teacher_proxy_comparison_performed": bool(comparison.get("teacher_proxy_comparison_performed")),
        "teacher_proxy_pearson_mean": float(comparison.get("teacher_proxy_pearson_mean", 0.0)),
        "teacher_proxy_spearman_mean": float(comparison.get("teacher_proxy_spearman_mean", 0.0)),
        "teacher_proxy_topk_overlap": comparison.get("topk_overlap") or {},
        "teacher_topk_utility_performed": bool(topk.get("teacher_topk_utility_performed")),
        "teacher_topk_val_table": topk.get("per_seed_val_table") or [],
        "teacher_beats_current_fraction": float(topk.get("teacher_beats_current_fraction", 0.0)),
        "teacher_beats_random_fraction": float(topk.get("teacher_beats_random_fraction", 0.0)),
        "teacher_beats_proxy_fraction": float(topk.get("teacher_beats_proxy_fraction", 0.0)),
        "teacher_mean_improvement_over_proxy": float(topk.get("teacher_mean_improvement_over_proxy", 0.0)),
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "videomae_training_performed": bool(train_summary.get("videomae_training_performed")),
        "context_teacher_large_training_performed": bool(train_summary.get("context_teacher_large_training_performed")),
        "selector_training_performed": bool(train_summary.get("selector_training_performed")) or bool(topk.get("selector_training_performed")),
        "current_importance_training_performed": bool(train_summary.get("current_importance_training_performed")) or bool(topk.get("current_importance_training_performed")),
        "token_extraction_performed": bool(train_summary.get("token_extraction_performed")) or bool(label_summary.get("token_extraction_performed")),
        "proxy_importance_regeneration_performed": bool(train_summary.get("proxy_importance_regeneration_performed")) or bool(label_summary.get("proxy_importance_regeneration_performed")),
        "new_tfds_shard_downloaded": bool(train_summary.get("new_tfds_shard_downloaded")),
        "model_download_performed": bool(train_summary.get("model_download_performed")),
        "data_token_shards_written": False,
        "data_importance_shards_written": bool(label_summary.get("data_importance_shards_written")),
        "checkpoint_saved": bool(train_summary.get("checkpoint_saved")) or bool(topk.get("checkpoint_saved")),
        "recommended_step31": decision.get("recommended_step31"),
        "teacher_decision": decision,
        "safety_gate_pass": safety_gate_pass,
        "reason": train_summary.get("reason") or label_summary.get("reason") or topk.get("reason"),
    }
    eval_json = Path(output["eval_json"])
    eval_md = Path(output["eval_md"])
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    eval_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    report = _format_report(result)
    eval_md.write_text(report, encoding="utf-8")
    if resolved_config == DEFAULT_CONFIG.resolve():
        STEP_DOC.write_text(_format_step_doc(result), encoding="utf-8")
        REPORT_DOC.write_text(report, encoding="utf-8")
        COMPARISON_DOC.write_text(_format_comparison_doc(result), encoding="utf-8")
        DECISION_DOC.write_text(_format_decision_doc(result), encoding="utf-8")
    return result


def _pass(
    train_summary: dict[str, Any],
    label_summary: dict[str, Any],
    comparison: dict[str, Any],
    topk: dict[str, Any],
    decision: dict[str, Any],
    safety_gate_pass: bool,
) -> bool:
    return (
        safety_gate_pass
        and bool(train_summary.get("trained_predictor_teacher_training_performed"))
        and bool(train_summary.get("teacher_val_loss_finite"))
        and bool(label_summary.get("teacher_occlusion_importance_generated"))
        and int(label_summary.get("num_samples") or 0) >= 64
        and list(label_summary.get("context_importance_shape_example") or []) == [16, 392]
        and bool(comparison.get("teacher_label_nontrivial"))
        and bool(comparison.get("teacher_proxy_comparison_performed"))
        and bool(topk.get("teacher_topk_utility_performed"))
        and bool(decision.get("recommended_step31"))
        and not bool(decision.get("context_utility_claim_allowed"))
        and not bool(decision.get("selector_training_allowed"))
        and not bool(decision.get("current_importance_training_allowed"))
    )


def _safety_gate_pass(
    train_summary: dict[str, Any],
    label_summary: dict[str, Any],
    comparison: dict[str, Any],
    topk: dict[str, Any],
    decision: dict[str, Any],
) -> bool:
    forbidden = [
        train_summary.get("new_tfds_shard_downloaded"),
        train_summary.get("model_download_performed"),
        train_summary.get("token_extraction_performed"),
        train_summary.get("proxy_importance_regeneration_performed"),
        label_summary.get("token_extraction_performed"),
        label_summary.get("proxy_importance_regeneration_performed"),
        train_summary.get("videomae_training_performed"),
        train_summary.get("context_teacher_large_training_performed"),
        train_summary.get("selector_training_performed"),
        topk.get("selector_training_performed"),
        train_summary.get("current_importance_training_performed"),
        topk.get("current_importance_training_performed"),
        label_summary.get("data_importance_shards_written"),
        train_summary.get("checkpoint_saved"),
        topk.get("checkpoint_saved"),
        decision.get("context_utility_claim_allowed"),
        decision.get("selector_training_allowed"),
        decision.get("current_importance_training_allowed"),
    ]
    return not any(bool(value) for value in forbidden) and bool(comparison.get("safety_gate_pass", True))


def _format_step_doc(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step30B BridgeData V2 TFDS Occlusion Teacher",
            "",
            "Step30B follows the Step30A finding that proxy topK is stable while full context remains noisy.",
            "",
            "- trains only a small current-conditioned predictor teacher",
            "- freezes the teacher and generates context-token occlusion delta labels",
            "- compares teacher labels to Step30A proxy labels",
            "- optionally runs teacher_topK tiny utility sanity",
            "- does not train selector or current importance",
            "- does not claim final context utility",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- teacher_beats_proxy_fraction: `{result['teacher_beats_proxy_fraction']}`",
            "",
        ]
    )


def _format_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 TFDS Occlusion Teacher Report",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- safe_stop: `{str(result['safe_stop']).lower()}`",
            f"- trained_predictor_teacher_training_performed: `{str(result['trained_predictor_teacher_training_performed']).lower()}`",
            f"- teacher_training_scope: `{result['teacher_training_scope']}`",
            f"- split_seeds_trained: `{result['split_seeds_trained']}`",
            f"- teacher_val_loss_finite: `{str(result['teacher_val_loss_finite']).lower()}`",
            f"- teacher_occlusion_importance_generated: `{str(result['teacher_occlusion_importance_generated']).lower()}`",
            f"- num_samples: `{result['num_samples']}`",
            f"- teacher_label_shape: `{result['teacher_label_shape']}`",
            f"- fallback_used_count: `{result['fallback_used_count']}`",
            f"- teacher_label_nontrivial: `{str(result['teacher_label_nontrivial']).lower()}`",
            f"- teacher_proxy_pearson_mean: `{result['teacher_proxy_pearson_mean']}`",
            f"- teacher_proxy_spearman_mean: `{result['teacher_proxy_spearman_mean']}`",
            f"- teacher_beats_current_fraction: `{result['teacher_beats_current_fraction']}`",
            f"- teacher_beats_random_fraction: `{result['teacher_beats_random_fraction']}`",
            f"- teacher_beats_proxy_fraction: `{result['teacher_beats_proxy_fraction']}`",
            f"- context_utility_claim_allowed: `{str(result['context_utility_claim_allowed']).lower()}`",
            f"- selector_training_allowed: `{str(result['selector_training_allowed']).lower()}`",
            f"- current_importance_training_allowed: `{str(result['current_importance_training_allowed']).lower()}`",
            f"- safety_gate_pass: `{str(result['safety_gate_pass']).lower()}`",
            "",
            "## Teacher Train Summary",
            "",
            "```json",
            json.dumps(result["teacher_seed_summaries"], indent=2, sort_keys=True),
            "```",
            "",
            "## Teacher TopK Utility",
            "",
            "```json",
            json.dumps(result["teacher_topk_val_table"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _format_comparison_doc(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 TFDS Teacher Proxy Comparison",
            "",
            f"- teacher_proxy_pearson_mean: `{result['teacher_proxy_pearson_mean']}`",
            f"- teacher_proxy_spearman_mean: `{result['teacher_proxy_spearman_mean']}`",
            f"- topk_overlap: `{result['teacher_proxy_topk_overlap']}`",
            f"- teacher_label_nontrivial: `{str(result['teacher_label_nontrivial']).lower()}`",
            "",
            "Low teacher/proxy correlation is not a failure; it can indicate a distinct trained-teacher label.",
            "",
        ]
    )


def _format_decision_doc(result: dict[str, Any]) -> str:
    recommended = result.get("recommended_step31") or {}
    return "\n".join(
        [
            "# BridgeData V2 TFDS Occlusion Teacher Decision",
            "",
            "Recommended Step31:",
            str(recommended.get("name")),
            "",
            "Do not claim final context utility.",
            "Do not train current importance yet.",
            "Do not train selector yet.",
            "",
            f"- condition: `{recommended.get('condition')}`",
            f"- scope: `{recommended.get('scope')}`",
            f"- context_utility_claim_allowed: `{str(result['context_utility_claim_allowed']).lower()}`",
            f"- selector_training_allowed: `{str(result['selector_training_allowed']).lower()}`",
            f"- current_importance_training_allowed: `{str(result['current_importance_training_allowed']).lower()}`",
            "",
        ]
    )


def _read_json_or_empty(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


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
    print(json.dumps(evaluate_step30b_occlusion_teacher(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
