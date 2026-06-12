"""Evaluate Step28 context sanity outputs and write decision reports."""

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

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_context_sanity_step28.yaml"
STEP_DOC = PROJECT_ROOT / "docs" / "STEP28_BRIDGEDATA_V2_TFDS_CONTEXT_SANITY.md"
REPORT_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TFDS_CONTEXT_SANITY_REPORT.md"
TEACHER_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TFDS_STRONGER_TEACHER_PLAN.md"
DECISION_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TFDS_NEXT_STEP_DECISION.md"


def evaluate_step28_context_sanity(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved_config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    output = config["output"]
    summary = _read_json_or_empty(output["sanity_summary_json"])
    context_utility = _read_json_or_empty(output["context_utility_json"])
    memorization = _read_json_or_empty(output["memorization_risk_json"])
    importance_label = _read_json_or_empty(output["importance_label_json"])
    teacher_plan = _read_json_or_empty(output["teacher_plan_json"])
    next_step = _read_json_or_empty(output["next_step_decision_json"])

    safe_stop = bool(summary.get("safe_stop", not summary))
    safety_gate_pass = _safety_gate_pass(summary)
    passed = _pass(summary, next_step, safe_stop, safety_gate_pass)
    result = {
        "stage": config["stage"],
        "pass": passed,
        "safe_stop": safe_stop,
        "context_sanity_performed": bool(summary.get("context_sanity_performed")),
        "step24_pass": bool(summary.get("step24_pass")),
        "step25_pass": bool(summary.get("step25_pass")),
        "step26_pass": bool(summary.get("step26_pass")),
        "step27_pass": bool(summary.get("step27_pass")),
        "num_samples": int(summary.get("num_samples") or 0),
        "current_only_can_overfit": bool(summary.get("current_only_can_overfit")),
        "current_only_relative_loss_decrease": summary.get("current_only_relative_loss_decrease"),
        "context_utility_claim_allowed": bool(summary.get("context_utility_claim_allowed")),
        "memorization_risk": summary.get("memorization_risk"),
        "proxy_importance_mass_advantage_over_random": summary.get(
            "proxy_importance_mass_advantage_over_random"
        ),
        "importance_label_is_proxy_only": bool(summary.get("importance_label_is_proxy_only")),
        "label_quality_note": importance_label.get("label_quality_note"),
        "train_current_importance_now": bool(summary.get("train_current_importance_now")),
        "recommended_step29": next_step.get(
            "recommended_step29",
            {
                "name": "BridgeData V2 TFDS 32/64-window train-val context utility sanity",
                "condition": "Step28 shows 4-sample overfit is memorization-prone",
                "scope": "use existing shard first; allow limited token extraction expansion, not full training",
            },
        ),
        "alternative_step29": next_step.get(
            "alternative_step29",
            {
                "name": "trained-predictor occlusion teacher dry-run",
                "condition": "if user chooses teacher-first path",
                "scope": "no selector training yet",
            },
        ),
        "teacher_plan": teacher_plan,
        "context_utility": context_utility,
        "memorization": memorization,
        "importance_label": importance_label,
        "download_performed": bool(summary.get("download_performed")),
        "model_download_performed": bool(summary.get("model_download_performed")),
        "training_performed": bool(summary.get("training_performed")),
        "optimizer_step_performed": bool(summary.get("optimizer_step_performed")),
        "token_extraction_performed": bool(summary.get("token_extraction_performed")),
        "importance_generation_performed": bool(summary.get("importance_generation_performed")),
        "videomae_training_performed": bool(summary.get("videomae_training_performed")),
        "teacher_training_performed": bool(summary.get("teacher_training_performed")),
        "selector_training_performed": bool(summary.get("selector_training_performed")),
        "current_importance_training_performed": bool(summary.get("current_importance_training_performed")),
        "world_model_training_performed": bool(summary.get("world_model_training_performed")),
        "data_token_shards_written": bool(summary.get("data_token_shards_written")),
        "data_importance_shards_written": bool(summary.get("data_importance_shards_written")),
        "checkpoint_saved": bool(summary.get("checkpoint_saved")),
        "reason": summary.get("reason"),
        "safety_gate_pass": safety_gate_pass,
    }

    eval_json = Path(output["eval_json"])
    eval_md = Path(output["eval_md"])
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    eval_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    eval_report = _format_eval_report(result)
    eval_md.write_text(eval_report, encoding="utf-8")
    if resolved_config_path == DEFAULT_CONFIG.resolve():
        _write_default_docs(result)
    return result


def _pass(summary: dict[str, Any], next_step: dict[str, Any], safe_stop: bool, safety_gate_pass: bool) -> bool:
    return (
        not safe_stop
        and safety_gate_pass
        and bool(summary.get("context_sanity_performed"))
        and bool(summary.get("step24_pass"))
        and bool(summary.get("step25_pass"))
        and bool(summary.get("step26_pass"))
        and bool(summary.get("step27_pass"))
        and bool(next_step.get("recommended_step29"))
        and not bool(summary.get("download_performed"))
        and not bool(summary.get("training_performed"))
        and not bool(summary.get("optimizer_step_performed"))
        and not bool(summary.get("token_extraction_performed"))
        and not bool(summary.get("importance_generation_performed"))
        and not bool(summary.get("data_token_shards_written"))
        and not bool(summary.get("data_importance_shards_written"))
        and not bool(summary.get("train_current_importance_now"))
        and not bool(summary.get("current_importance_training_performed"))
        and not bool(summary.get("context_utility_claim_allowed"))
    )


def _safety_gate_pass(summary: dict[str, Any]) -> bool:
    return (
        bool(summary.get("safety_gate_pass", True))
        and not bool(summary.get("download_performed"))
        and not bool(summary.get("model_download_performed"))
        and not bool(summary.get("training_performed"))
        and not bool(summary.get("optimizer_step_performed"))
        and not bool(summary.get("token_extraction_performed"))
        and not bool(summary.get("importance_generation_performed"))
        and not bool(summary.get("videomae_training_performed"))
        and not bool(summary.get("teacher_training_performed"))
        and not bool(summary.get("selector_training_performed"))
        and not bool(summary.get("current_importance_training_performed"))
        and not bool(summary.get("world_model_training_performed"))
        and not bool(summary.get("data_token_shards_written"))
        and not bool(summary.get("data_importance_shards_written"))
        and not bool(summary.get("checkpoint_saved"))
    )


def _write_default_docs(result: dict[str, Any]) -> None:
    STEP_DOC.write_text(_format_step_doc(result), encoding="utf-8")
    REPORT_DOC.write_text(_format_context_report(result), encoding="utf-8")
    TEACHER_DOC.write_text(_format_teacher_plan(result), encoding="utf-8")
    DECISION_DOC.write_text(_format_decision_doc(result), encoding="utf-8")


def _format_eval_report(result: dict[str, Any]) -> str:
    return _format_context_report(result)


def _format_step_doc(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step28 BridgeData V2 TFDS Context Sanity",
            "",
            "Step28 is a read-only diagnosis and decision step after the Step27 tiny overfit run.",
            "",
            "## Why Step28 Exists",
            "",
            "- Step27 proved that a tiny predictor can overfit 4 real BridgeData token samples.",
            "- Step27 did not prove final model performance or context bottleneck utility.",
            "- The current_only policy also overfit, so 4 samples are too small to support a context advantage claim.",
            "- Step28 keeps current tokens full and does not train current importance.",
            "",
            "## Guardrail Result",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- safe_stop: `{str(result['safe_stop']).lower()}`",
            f"- context_utility_claim_allowed: `{str(result['context_utility_claim_allowed']).lower()}`",
            f"- train_current_importance_now: `{str(result['train_current_importance_now']).lower()}`",
            "",
        ]
    )


def _format_context_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 TFDS Context Sanity Report",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- safe_stop: `{str(result['safe_stop']).lower()}`",
            f"- context_sanity_performed: `{str(result['context_sanity_performed']).lower()}`",
            f"- step24_pass: `{str(result['step24_pass']).lower()}`",
            f"- step25_pass: `{str(result['step25_pass']).lower()}`",
            f"- step26_pass: `{str(result['step26_pass']).lower()}`",
            f"- step27_pass: `{str(result['step27_pass']).lower()}`",
            f"- num_samples: `{result['num_samples']}`",
            f"- current_only_can_overfit: `{str(result['current_only_can_overfit']).lower()}`",
            f"- current_only_relative_loss_decrease: `{result['current_only_relative_loss_decrease']}`",
            f"- memorization_risk: `{result['memorization_risk']}`",
            f"- proxy_importance_mass_advantage_over_random: `{result['proxy_importance_mass_advantage_over_random']}`",
            f"- context_utility_claim_allowed: `{str(result['context_utility_claim_allowed']).lower()}`",
            f"- importance_label_is_proxy_only: `{str(result['importance_label_is_proxy_only']).lower()}`",
            f"- label_quality_note: `{result['label_quality_note']}`",
            f"- train_current_importance_now: `{str(result['train_current_importance_now']).lower()}`",
            f"- download_performed: `{str(result['download_performed']).lower()}`",
            f"- training_performed: `{str(result['training_performed']).lower()}`",
            f"- optimizer_step_performed: `{str(result['optimizer_step_performed']).lower()}`",
            f"- token_extraction_performed: `{str(result['token_extraction_performed']).lower()}`",
            f"- importance_generation_performed: `{str(result['importance_generation_performed']).lower()}`",
            f"- data_token_shards_written: `{str(result['data_token_shards_written']).lower()}`",
            f"- data_importance_shards_written: `{str(result['data_importance_shards_written']).lower()}`",
            f"- safety_gate_pass: `{str(result['safety_gate_pass']).lower()}`",
            "",
            "## Interpretation",
            "",
            "Step27 is best read as a tiny overfit sanity result. Because current_only also overfits almost perfectly, "
            "the experiment is memorization-prone and does not validate context utility. Step25 proxy importance shows "
            "ranking concentration, but it remains a proxy-only label.",
            "",
            "## Context Utility Diagnosis",
            "",
            "```json",
            json.dumps(result["context_utility"], indent=2, sort_keys=True),
            "```",
            "",
            "## Memorization Risk",
            "",
            "```json",
            json.dumps(result["memorization"], indent=2, sort_keys=True),
            "```",
            "",
            "## Importance Label Diagnosis",
            "",
            "```json",
            json.dumps(result["importance_label"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _format_teacher_plan(result: dict[str, Any]) -> str:
    teacher_plan = result.get("teacher_plan") or {}
    lines = [
        "# BridgeData V2 TFDS Stronger Teacher Plan",
        "",
        f"- recommended_primary_next_step: `{teacher_plan.get('recommended_primary_next_step')}`",
        f"- recommended_secondary_next_step: `{teacher_plan.get('recommended_secondary_next_step')}`",
        f"- do_not_train_current_importance_yet: `{str(teacher_plan.get('do_not_train_current_importance_yet')).lower()}`",
        "",
        "## Candidate Plans",
        "",
    ]
    for candidate in teacher_plan.get("candidate_plans", []):
        lines.extend(
            [
                f"### Candidate {candidate.get('id')}: {candidate.get('name')}",
                "",
                candidate.get("summary", ""),
                "",
                f"- recommended_stage: `{candidate.get('recommended_stage')}`",
                f"- allowed_now: `{str(candidate.get('allowed_now')).lower()}`",
                "",
            ]
        )
    return "\n".join(lines)


def _format_decision_doc(result: dict[str, Any]) -> str:
    next_step = result.get("recommended_step29") or {}
    alternative = result.get("alternative_step29") or {}
    return "\n".join(
        [
            "# BridgeData V2 TFDS Next Step Decision",
            "",
            "Recommended Step29:",
            "BridgeData V2 TFDS 32/64-window train-val context utility sanity using existing shard first.",
            "",
            "Do not train current importance yet.",
            "Do not train selector yet.",
            "Do not claim context utility from 4-sample overfit.",
            "",
            f"- recommended_step29_name: `{next_step.get('name')}`",
            f"- recommended_step29_condition: `{next_step.get('condition')}`",
            f"- recommended_step29_scope: `{next_step.get('scope')}`",
            f"- alternative_step29_name: `{alternative.get('name')}`",
            f"- alternative_step29_condition: `{alternative.get('condition')}`",
            f"- alternative_step29_scope: `{alternative.get('scope')}`",
            f"- context_utility_claim_allowed: `{str(result['context_utility_claim_allowed']).lower()}`",
            f"- train_current_importance_now: `{str(result['train_current_importance_now']).lower()}`",
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
    print(json.dumps(evaluate_step28_context_sanity(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

