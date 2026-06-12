"""Evaluate Step29 train/val context utility sanity outputs."""

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

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_trainval_context_utility_step29.yaml"
STEP_DOC = PROJECT_ROOT / "docs" / "STEP29_BRIDGEDATA_V2_TFDS_TRAINVAL_CONTEXT_UTILITY.md"
REPORT_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TFDS_TRAINVAL_CONTEXT_UTILITY_REPORT.md"
DECISION_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TFDS_TRAINVAL_CONTEXT_UTILITY_DECISION.md"


def evaluate_step29_trainval_context_utility(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved_config = Path(config_path).resolve()
    config = _load_yaml(config_path)
    output = config["output"]
    selected_count = _count_jsonl(output["selected_windows_jsonl"])
    split = _read_json_or_empty(output["split_json"])
    token_summary = _read_json_or_empty(output["token_summary_json"])
    importance_summary = _read_json_or_empty(output["importance_summary_json"])
    trainval_summary = _read_json_or_empty(output["trainval_summary_json"])
    policy_comparison = _read_json_or_empty(output["policy_comparison_json"])
    decision = _read_json_or_empty(output["context_utility_decision_json"])
    safe_stop = bool(trainval_summary.get("safe_stop", not trainval_summary))
    safety_gate_pass = _safety_gate_pass(token_summary, importance_summary, trainval_summary)
    passed = _pass(selected_count, split, token_summary, importance_summary, trainval_summary, decision, safety_gate_pass)
    result = {
        "stage": config["stage"],
        "pass": passed,
        "safe_stop": safe_stop,
        "num_selected_windows": selected_count,
        "num_train_windows": int(split.get("num_train_windows") or trainval_summary.get("num_train_windows") or 0),
        "num_val_windows": int(split.get("num_val_windows") or trainval_summary.get("num_val_windows") or 0),
        "train_val_trajectory_disjoint": bool(split.get("train_val_trajectory_disjoint")),
        "limited_token_extraction_performed": bool(token_summary.get("limited_token_extraction_performed")),
        "token_shapes": {
            "context": token_summary.get("context_token_shape_example"),
            "current": token_summary.get("current_token_shape_example"),
            "future": token_summary.get("future_token_shape_example"),
        },
        "limited_proxy_importance_generation_performed": bool(
            importance_summary.get("limited_proxy_importance_generation_performed")
        ),
        "importance_shapes": {
            "context": importance_summary.get("context_importance_shape_example"),
            "temporal": importance_summary.get("temporal_importance_shape_example"),
            "spatial": importance_summary.get("spatial_importance_shape_example"),
        },
        "tiny_trainval_training_performed": bool(trainval_summary.get("tiny_trainval_training_performed")),
        "optimizer_step_performed": bool(trainval_summary.get("optimizer_step_performed")),
        "optimizer_step_scope": trainval_summary.get("optimizer_step_scope"),
        "policies_trained": trainval_summary.get("policies_trained") or [],
        "policy_metric_table": policy_comparison.get("policy_metrics") or trainval_summary.get("policy_metric_table") or [],
        "context_utility_sanity_signal": decision.get(
            "context_utility_sanity_signal",
            trainval_summary.get("context_utility_sanity_signal"),
        ),
        "context_utility_claim_allowed": False,
        "current_tokens_kept_full": bool(trainval_summary.get("current_tokens_kept_full", True)),
        "train_current_importance": bool(trainval_summary.get("train_current_importance", False)),
        "videomae_training_performed": bool(trainval_summary.get("videomae_training_performed")),
        "teacher_training_performed": bool(trainval_summary.get("teacher_training_performed")),
        "selector_training_performed": bool(trainval_summary.get("selector_training_performed")),
        "current_importance_training_performed": bool(
            trainval_summary.get("current_importance_training_performed")
        ),
        "new_tfds_shard_downloaded": bool(trainval_summary.get("new_tfds_shard_downloaded")),
        "model_download_performed": bool(trainval_summary.get("model_download_performed"))
        or bool(token_summary.get("model_download_performed")),
        "data_token_shards_written": bool(trainval_summary.get("data_token_shards_written"))
        or bool(token_summary.get("data_token_shards_written")),
        "data_importance_shards_written": bool(trainval_summary.get("data_importance_shards_written"))
        or bool(importance_summary.get("data_importance_shards_written")),
        "checkpoint_saved": bool(trainval_summary.get("checkpoint_saved")),
        "recommended_step30": decision.get("recommended_step30"),
        "context_utility_decision": decision,
        "policy_comparison": policy_comparison,
        "safety_gate_pass": safety_gate_pass,
        "reason": trainval_summary.get("reason"),
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
        DECISION_DOC.write_text(_format_decision_doc(result), encoding="utf-8")
    return result


def _pass(
    selected_count: int,
    split: dict[str, Any],
    token_summary: dict[str, Any],
    importance_summary: dict[str, Any],
    trainval_summary: dict[str, Any],
    decision: dict[str, Any],
    safety_gate_pass: bool,
) -> bool:
    return (
        safety_gate_pass
        and selected_count >= 32
        and int(split.get("num_val_windows") or 0) >= 8
        and bool(token_summary.get("limited_token_extraction_performed"))
        and bool(importance_summary.get("limited_proxy_importance_generation_performed"))
        and bool(trainval_summary.get("tiny_trainval_training_performed"))
        and bool(trainval_summary.get("optimizer_step_performed"))
        and trainval_summary.get("optimizer_step_scope") == "tiny_world_model_predictor_only"
        and bool(trainval_summary.get("all_val_losses_finite"))
        and bool(decision.get("context_utility_sanity_signal"))
        and not bool(decision.get("context_utility_claim_allowed"))
    )


def _safety_gate_pass(
    token_summary: dict[str, Any],
    importance_summary: dict[str, Any],
    trainval_summary: dict[str, Any],
) -> bool:
    forbidden = [
        trainval_summary.get("new_tfds_shard_downloaded"),
        token_summary.get("new_tfds_shard_downloaded"),
        trainval_summary.get("model_download_performed"),
        token_summary.get("model_download_performed"),
        trainval_summary.get("videomae_training_performed"),
        trainval_summary.get("teacher_training_performed"),
        trainval_summary.get("selector_training_performed"),
        trainval_summary.get("current_importance_training_performed"),
        trainval_summary.get("data_token_shards_written"),
        token_summary.get("data_token_shards_written"),
        trainval_summary.get("data_importance_shards_written"),
        importance_summary.get("data_importance_shards_written"),
        trainval_summary.get("checkpoint_saved"),
    ]
    return not any(bool(value) for value in forbidden)


def _format_step_doc(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step29 BridgeData V2 TFDS Train-Val Context Utility",
            "",
            "Step29 follows the Step28 finding that 4-sample overfit is memorization-prone.",
            "",
            "- uses existing TFDS shard only",
            "- expands to a bounded 32-window train/val sanity run",
            "- performs limited VideoMAE token extraction and proxy importance generation",
            "- trains only the tiny world-model predictor",
            "- does not train selector or current importance",
            "- does not claim final context utility",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- context_utility_sanity_signal: `{result['context_utility_sanity_signal']}`",
            "",
        ]
    )


def _format_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 TFDS Train-Val Context Utility Report",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- safe_stop: `{str(result['safe_stop']).lower()}`",
            f"- num_selected_windows: `{result['num_selected_windows']}`",
            f"- num_train_windows: `{result['num_train_windows']}`",
            f"- num_val_windows: `{result['num_val_windows']}`",
            f"- train_val_trajectory_disjoint: `{str(result['train_val_trajectory_disjoint']).lower()}`",
            f"- limited_token_extraction_performed: `{str(result['limited_token_extraction_performed']).lower()}`",
            f"- token_shapes: `{result['token_shapes']}`",
            f"- limited_proxy_importance_generation_performed: `{str(result['limited_proxy_importance_generation_performed']).lower()}`",
            f"- importance_shapes: `{result['importance_shapes']}`",
            f"- tiny_trainval_training_performed: `{str(result['tiny_trainval_training_performed']).lower()}`",
            f"- optimizer_step_performed: `{str(result['optimizer_step_performed']).lower()}`",
            f"- optimizer_step_scope: `{result['optimizer_step_scope']}`",
            f"- context_utility_sanity_signal: `{result['context_utility_sanity_signal']}`",
            f"- context_utility_claim_allowed: `{str(result['context_utility_claim_allowed']).lower()}`",
            f"- current_tokens_kept_full: `{str(result['current_tokens_kept_full']).lower()}`",
            f"- train_current_importance: `{str(result['train_current_importance']).lower()}`",
            f"- new_tfds_shard_downloaded: `{str(result['new_tfds_shard_downloaded']).lower()}`",
            f"- model_download_performed: `{str(result['model_download_performed']).lower()}`",
            f"- data_token_shards_written: `{str(result['data_token_shards_written']).lower()}`",
            f"- data_importance_shards_written: `{str(result['data_importance_shards_written']).lower()}`",
            f"- safety_gate_pass: `{str(result['safety_gate_pass']).lower()}`",
            "",
            "## Policy Train/Val Table",
            "",
            "```json",
            json.dumps(result["policy_metric_table"], indent=2, sort_keys=True),
            "```",
            "",
            "## Context Utility Decision",
            "",
            "```json",
            json.dumps(result["context_utility_decision"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _format_decision_doc(result: dict[str, Any]) -> str:
    signal = str(result["context_utility_sanity_signal"])
    recommended = result.get("recommended_step30") or {}
    if signal == "positive":
        headline = "Recommended Step30:\ntrained-predictor occlusion teacher on expanded train split."
    elif signal == "negative_or_current_dominant":
        headline = "Recommended Step30:\ndiagnose horizon/current dominance and reconsider context objective before selector training."
    else:
        headline = "Recommended Step30:\nimprove teacher label or increase window diversity before selector training."
    return "\n".join(
        [
            "# BridgeData V2 TFDS Train-Val Context Utility Decision",
            "",
            headline,
            "",
            "Do not claim final context utility.",
            "Do not train current importance yet.",
            "Do not train selector yet.",
            "",
            f"- context_utility_sanity_signal: `{signal}`",
            f"- context_utility_claim_allowed: `{str(result['context_utility_claim_allowed']).lower()}`",
            f"- recommended_step30_name: `{recommended.get('name')}`",
            f"- recommended_step30_condition: `{recommended.get('condition')}`",
            f"- recommended_step30_scope: `{recommended.get('scope')}`",
            "",
        ]
    )


def _count_jsonl(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


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
    print(json.dumps(evaluate_step29_trainval_context_utility(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

