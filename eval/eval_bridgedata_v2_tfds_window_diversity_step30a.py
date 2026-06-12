"""Evaluate Step30A BridgeData TFDS window diversity outputs."""

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

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_window_diversity_step30a.yaml"
STEP_DOC = PROJECT_ROOT / "docs" / "STEP30A_BRIDGEDATA_V2_TFDS_WINDOW_DIVERSITY.md"
REPORT_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TFDS_WINDOW_DIVERSITY_REPORT.md"
DECISION_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TFDS_WINDOW_DIVERSITY_DECISION.md"


def evaluate_step30a_window_diversity(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved_config = Path(config_path).resolve()
    config = _load_yaml(config_path)
    output = config["output"]
    selection = _read_json_or_empty(output["selection_summary_json"])
    splits = _read_json_or_empty(output["multiseed_splits_json"])
    token_summary = _read_json_or_empty(output["token_summary_json"])
    importance_summary = _read_json_or_empty(output["importance_summary_json"])
    trainval_runs = _read_json_or_empty(output["trainval_runs_json"])
    stability = _read_json_or_empty(output["stability_summary_json"])
    decision = _read_json_or_empty(output["context_signal_decision_json"])
    safety_gate_pass = _safety_gate_pass(token_summary, importance_summary, trainval_runs, decision)
    safe_stop = bool(
        selection.get("safe_stop")
        or splits.get("safe_stop")
        or token_summary.get("safe_stop")
        or importance_summary.get("safe_stop")
        or trainval_runs.get("safe_stop")
    )
    selected_count = int(selection.get("num_selected_windows") or _count_jsonl(output["selected_windows_jsonl"]))
    result = {
        "stage": config["stage"],
        "pass": _pass(selected_count, splits, token_summary, importance_summary, trainval_runs, decision, safety_gate_pass),
        "safe_stop": safe_stop,
        "num_selected_windows": selected_count,
        "num_selected_trajectories": int(selection.get("num_selected_trajectories") or 0),
        "trajectory_window_counts": selection.get("trajectory_window_counts") or {},
        "num_splits": int(splits.get("num_splits") or trainval_runs.get("num_splits") or 0),
        "split_seeds": splits.get("seeds") or trainval_runs.get("split_seeds") or [],
        "split_summaries": _split_summaries(splits),
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
        "tiny_trainval_training_performed": bool(trainval_runs.get("tiny_trainval_training_performed")),
        "optimizer_step_performed": bool(trainval_runs.get("optimizer_step_performed")),
        "optimizer_step_scope": trainval_runs.get("optimizer_step_scope"),
        "policies_trained": trainval_runs.get("policies_trained") or [],
        "per_seed_val_table": _per_seed_val_table(trainval_runs),
        "proxy_beats_current_fraction": float(stability.get("proxy_beats_current_fraction", 0.0)),
        "proxy_beats_random_fraction": float(stability.get("proxy_beats_random_fraction", 0.0)),
        "full_beats_current_fraction": float(stability.get("full_beats_current_fraction", 0.0)),
        "mean_proxy_improvement_over_current": float(stability.get("mean_proxy_improvement_over_current", 0.0)),
        "mean_proxy_improvement_over_random": float(stability.get("mean_proxy_improvement_over_random", 0.0)),
        "context_signal_stability": decision.get(
            "context_signal_stability",
            stability.get("context_signal_stability", trainval_runs.get("context_signal_stability")),
        ),
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "new_tfds_shard_downloaded": bool(trainval_runs.get("new_tfds_shard_downloaded")),
        "model_download_performed": bool(trainval_runs.get("model_download_performed"))
        or bool(token_summary.get("model_download_performed")),
        "videomae_training_performed": bool(trainval_runs.get("videomae_training_performed")),
        "teacher_training_performed": bool(trainval_runs.get("teacher_training_performed")),
        "selector_training_performed": bool(trainval_runs.get("selector_training_performed")),
        "current_importance_training_performed": bool(trainval_runs.get("current_importance_training_performed")),
        "data_token_shards_written": bool(trainval_runs.get("data_token_shards_written"))
        or bool(token_summary.get("data_token_shards_written")),
        "data_importance_shards_written": bool(trainval_runs.get("data_importance_shards_written"))
        or bool(importance_summary.get("data_importance_shards_written")),
        "checkpoint_saved": bool(trainval_runs.get("checkpoint_saved")),
        "recommended_step30b": decision.get("recommended_step30b"),
        "context_signal_decision": decision,
        "stability_summary": stability,
        "safety_gate_pass": safety_gate_pass,
        "reason": trainval_runs.get("reason") or token_summary.get("reason") or importance_summary.get("reason"),
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
    splits: dict[str, Any],
    token_summary: dict[str, Any],
    importance_summary: dict[str, Any],
    trainval_runs: dict[str, Any],
    decision: dict[str, Any],
    safety_gate_pass: bool,
) -> bool:
    return (
        safety_gate_pass
        and selected_count >= 64
        and int(splits.get("num_splits") or 0) >= 2
        and bool(token_summary.get("limited_token_extraction_performed"))
        and bool(importance_summary.get("limited_proxy_importance_generation_performed"))
        and bool(trainval_runs.get("tiny_trainval_training_performed"))
        and bool(trainval_runs.get("optimizer_step_performed"))
        and trainval_runs.get("optimizer_step_scope") == "tiny_world_model_predictor_only"
        and bool(trainval_runs.get("all_val_losses_finite"))
        and bool(decision.get("context_signal_stability"))
        and not bool(decision.get("context_utility_claim_allowed"))
        and not bool(decision.get("selector_training_allowed"))
        and not bool(decision.get("current_importance_training_allowed"))
    )


def _safety_gate_pass(
    token_summary: dict[str, Any],
    importance_summary: dict[str, Any],
    trainval_runs: dict[str, Any],
    decision: dict[str, Any],
) -> bool:
    forbidden = [
        trainval_runs.get("new_tfds_shard_downloaded"),
        token_summary.get("new_tfds_shard_downloaded"),
        trainval_runs.get("model_download_performed"),
        token_summary.get("model_download_performed"),
        trainval_runs.get("videomae_training_performed"),
        trainval_runs.get("teacher_training_performed"),
        trainval_runs.get("selector_training_performed"),
        trainval_runs.get("current_importance_training_performed"),
        trainval_runs.get("data_token_shards_written"),
        token_summary.get("data_token_shards_written"),
        trainval_runs.get("data_importance_shards_written"),
        importance_summary.get("data_importance_shards_written"),
        trainval_runs.get("checkpoint_saved"),
        decision.get("context_utility_claim_allowed"),
        decision.get("selector_training_allowed"),
        decision.get("current_importance_training_allowed"),
    ]
    return not any(bool(value) for value in forbidden)


def _split_summaries(splits: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "split_seed": int(split.get("split_seed", -1)),
            "num_train_windows": int(split.get("num_train_windows", 0)),
            "num_val_windows": int(split.get("num_val_windows", 0)),
            "trajectory_disjoint": bool(split.get("train_val_trajectory_disjoint")),
            "fallback_used": bool(split.get("fallback_used")),
            "fallback_reason": split.get("fallback_reason"),
        }
        for split in splits.get("splits", [])
    ]


def _per_seed_val_table(trainval_runs: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in trainval_runs.get("runs", []):
        row = {
            "split_seed": int(run.get("split_seed", -1)),
            "num_train_windows": int(run.get("num_train_windows", 0)),
            "num_val_windows": int(run.get("num_val_windows", 0)),
            "trajectory_disjoint": bool(run.get("trajectory_disjoint")),
        }
        for metric in run.get("policy_metrics", []):
            row[str(metric["policy"])] = float(metric.get("val_final_loss"))
        rows.append(row)
    return rows


def _format_step_doc(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step30A BridgeData V2 TFDS Window Diversity",
            "",
            "Step30A follows the Step29 inconclusive train/val context utility sanity result.",
            "",
            "- uses the existing BridgeData V2 TFDS shard only",
            "- expands selected windows and trajectory coverage inside the existing Step23.5 manifest",
            "- runs multi-seed trajectory-disjoint train/val splits",
            "- performs limited token extraction and proxy importance generation",
            "- trains only the tiny world-model predictor",
            "- does not train selector or current importance",
            "- does not claim final context utility",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- context_signal_stability: `{result['context_signal_stability']}`",
            "",
        ]
    )


def _format_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 TFDS Window Diversity Report",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- safe_stop: `{str(result['safe_stop']).lower()}`",
            f"- num_selected_windows: `{result['num_selected_windows']}`",
            f"- num_selected_trajectories: `{result['num_selected_trajectories']}`",
            f"- split_seeds: `{result['split_seeds']}`",
            f"- limited_token_extraction_performed: `{str(result['limited_token_extraction_performed']).lower()}`",
            f"- token_shapes: `{result['token_shapes']}`",
            f"- limited_proxy_importance_generation_performed: `{str(result['limited_proxy_importance_generation_performed']).lower()}`",
            f"- importance_shapes: `{result['importance_shapes']}`",
            f"- tiny_trainval_training_performed: `{str(result['tiny_trainval_training_performed']).lower()}`",
            f"- optimizer_step_scope: `{result['optimizer_step_scope']}`",
            f"- proxy_beats_current_fraction: `{result['proxy_beats_current_fraction']}`",
            f"- proxy_beats_random_fraction: `{result['proxy_beats_random_fraction']}`",
            f"- full_beats_current_fraction: `{result['full_beats_current_fraction']}`",
            f"- context_signal_stability: `{result['context_signal_stability']}`",
            f"- context_utility_claim_allowed: `{str(result['context_utility_claim_allowed']).lower()}`",
            f"- selector_training_allowed: `{str(result['selector_training_allowed']).lower()}`",
            f"- current_importance_training_allowed: `{str(result['current_importance_training_allowed']).lower()}`",
            f"- new_tfds_shard_downloaded: `{str(result['new_tfds_shard_downloaded']).lower()}`",
            f"- model_download_performed: `{str(result['model_download_performed']).lower()}`",
            f"- data_token_shards_written: `{str(result['data_token_shards_written']).lower()}`",
            f"- data_importance_shards_written: `{str(result['data_importance_shards_written']).lower()}`",
            f"- safety_gate_pass: `{str(result['safety_gate_pass']).lower()}`",
            "",
            "## Split Summary",
            "",
            "```json",
            json.dumps(result["split_summaries"], indent=2, sort_keys=True),
            "```",
            "",
            "## Per-Seed Validation Table",
            "",
            "```json",
            json.dumps(result["per_seed_val_table"], indent=2, sort_keys=True),
            "```",
            "",
            "## Stability Summary",
            "",
            "```json",
            json.dumps(result["stability_summary"], indent=2, sort_keys=True),
            "```",
            "",
            "## Context Signal Decision",
            "",
            "```json",
            json.dumps(result["context_signal_decision"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _format_decision_doc(result: dict[str, Any]) -> str:
    signal = str(result["context_signal_stability"])
    recommended = result.get("recommended_step30b") or {}
    if signal in {"positive_but_not_final", "weak_proxy_positive_full_context_noisy"}:
        headline = "Recommended Step30B:\ntrained-predictor occlusion teacher on expanded windows."
    elif signal == "negative_or_current_dominant":
        headline = "Recommended Step30B:\ndiagnose current dominance / horizon / representation before selector training."
    else:
        headline = "Recommended Step30B:\nincrease data diversity or improve temporal tokenization before teacher/selector."
    return "\n".join(
        [
            "# BridgeData V2 TFDS Window Diversity Decision",
            "",
            headline,
            "",
            "Do not claim final context utility.",
            "Do not train current importance yet.",
            "Do not train selector yet.",
            "",
            f"- context_signal_stability: `{signal}`",
            f"- context_utility_claim_allowed: `{str(result['context_utility_claim_allowed']).lower()}`",
            f"- selector_training_allowed: `{str(result['selector_training_allowed']).lower()}`",
            f"- current_importance_training_allowed: `{str(result['current_importance_training_allowed']).lower()}`",
            f"- recommended_step30b_name: `{recommended.get('name')}`",
            f"- recommended_step30b_condition: `{recommended.get('condition')}`",
            f"- recommended_step30b_scope: `{recommended.get('scope')}`",
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
    print(json.dumps(evaluate_step30a_window_diversity(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
