"""Evaluate Step27 BridgeData TFDS tiny overfit outputs."""

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

from data.bridgedata_v2_tfds_world_model_training_metrics import LOSS_QUALITY_NOTE

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_world_model_tiny_overfit_step27.yaml"
DOC_REPORT = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TFDS_WORLD_MODEL_TINY_OVERFIT_REPORT.md"


def evaluate_step27_tiny_overfit(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved_config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    output = config["output"]
    training_summary = _read_json_or_empty(output["training_summary_json"])
    loss_curves = _read_json_or_empty(output["loss_curves_json"])
    policy_comparison = _read_json_or_empty(output["policy_comparison_json"])
    final_eval = _read_json_or_empty(output["final_eval_json"])
    safe_stop = bool(training_summary.get("safe_stop", not training_summary))
    safety_gate_pass = _safety_gate_pass(training_summary)
    passed = _pass(training_summary, policy_comparison, safe_stop, safety_gate_pass)
    result = {
        "stage": config["stage"],
        "pass": passed,
        "safe_stop": safe_stop,
        "tiny_overfit_training_performed": bool(training_summary.get("tiny_overfit_training_performed")),
        "training_performed": bool(training_summary.get("training_performed")),
        "tiny_overfit_training_only": bool(training_summary.get("tiny_overfit_training_only", True)),
        "num_samples": int(training_summary.get("num_samples") or 0),
        "policies_trained": training_summary.get("policies_trained") or [],
        "train_steps": int(training_summary.get("train_steps") or 0),
        "optimizer": training_summary.get("optimizer"),
        "optimizer_step_performed": bool(training_summary.get("optimizer_step_performed")),
        "optimizer_step_scope": training_summary.get("optimizer_step_scope"),
        "all_losses_finite": bool(training_summary.get("all_losses_finite")),
        "policies_with_loss_decrease": int(training_summary.get("policies_with_loss_decrease") or 0),
        "acceptance_pass": bool(training_summary.get("acceptance_pass")),
        "current_tokens_kept_full": bool(training_summary.get("current_tokens_kept_full")),
        "train_current_importance": bool(training_summary.get("train_current_importance")),
        "current_importance_training_performed": bool(training_summary.get("current_importance_training_performed")),
        "videomae_training_performed": bool(training_summary.get("videomae_training_performed")),
        "teacher_training_performed": bool(training_summary.get("teacher_training_performed")),
        "selector_training_performed": bool(training_summary.get("selector_training_performed")),
        "world_model_large_training_performed": bool(training_summary.get("world_model_large_training_performed")),
        "token_extraction_performed": bool(training_summary.get("token_extraction_performed")),
        "importance_generation_performed": bool(training_summary.get("importance_generation_performed")),
        "download_performed": bool(training_summary.get("download_performed")),
        "model_download_performed": bool(training_summary.get("model_download_performed")),
        "data_token_shards_written": bool(training_summary.get("data_token_shards_written")),
        "data_importance_shards_written": bool(training_summary.get("data_importance_shards_written")),
        "checkpoint_saved": bool(training_summary.get("checkpoint_saved")),
        "tiny_overfit_result_not_final_performance": bool(
            training_summary.get("tiny_overfit_result_not_final_performance")
        ),
        "policy_metric_table": policy_comparison.get("policy_metrics") or training_summary.get("policy_metric_table") or [],
        "policy_comparison": policy_comparison,
        "loss_curves": loss_curves,
        "final_eval": final_eval,
        "loss_quality_note": LOSS_QUALITY_NOTE,
        "reason": training_summary.get("reason"),
        "safety_gate_pass": safety_gate_pass,
        "recommended_step28": {
            "name": "BridgeData V2 TFDS context predictor sanity and stronger teacher planning",
            "condition": "tiny overfit passed",
            "scope": "decide whether to scale data or improve teacher labels",
        },
    }
    eval_json = Path(output["eval_json"])
    eval_md = Path(output["eval_md"])
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    eval_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    report = _format_report(result)
    eval_md.parent.mkdir(parents=True, exist_ok=True)
    eval_md.write_text(report, encoding="utf-8")
    if resolved_config_path == DEFAULT_CONFIG.resolve():
        DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
        DOC_REPORT.write_text(report, encoding="utf-8")
    return result


def _pass(
    summary: dict[str, Any],
    policy_comparison: dict[str, Any],
    safe_stop: bool,
    safety_gate_pass: bool,
) -> bool:
    return (
        not safe_stop
        and safety_gate_pass
        and bool(summary.get("tiny_overfit_training_performed"))
        and int(summary.get("num_samples") or 0) >= 1
        and bool(summary.get("optimizer_step_performed"))
        and summary.get("optimizer_step_scope") == "tiny_world_model_predictor_only"
        and bool(summary.get("all_losses_finite"))
        and bool(summary.get("acceptance_pass"))
        and int(policy_comparison.get("policies_with_loss_decrease") or 0) >= 3
        and bool(summary.get("current_tokens_kept_full"))
        and not bool(summary.get("train_current_importance"))
        and not bool(summary.get("current_importance_training_performed"))
        and not bool(summary.get("videomae_training_performed"))
        and not bool(summary.get("teacher_training_performed"))
        and not bool(summary.get("selector_training_performed"))
        and not bool(summary.get("world_model_large_training_performed"))
        and not bool(summary.get("token_extraction_performed"))
        and not bool(summary.get("importance_generation_performed"))
        and not bool(summary.get("download_performed"))
        and not bool(summary.get("data_token_shards_written"))
        and not bool(summary.get("data_importance_shards_written"))
        and not bool(summary.get("checkpoint_saved"))
    )


def _safety_gate_pass(summary: dict[str, Any]) -> bool:
    return (
        bool(summary.get("safety_gate_pass", True))
        and not bool(summary.get("download_performed"))
        and not bool(summary.get("model_download_performed"))
        and not bool(summary.get("videomae_training_performed"))
        and not bool(summary.get("teacher_training_performed"))
        and not bool(summary.get("selector_training_performed"))
        and not bool(summary.get("current_importance_training_performed"))
        and not bool(summary.get("world_model_large_training_performed"))
        and not bool(summary.get("token_extraction_performed"))
        and not bool(summary.get("importance_generation_performed"))
        and not bool(summary.get("data_token_shards_written"))
        and not bool(summary.get("data_importance_shards_written"))
        and not bool(summary.get("checkpoint_saved"))
    )


def _format_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 TFDS World-Model Tiny Overfit Report",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- safe_stop: `{str(result['safe_stop']).lower()}`",
            f"- tiny_overfit_training_performed: `{str(result['tiny_overfit_training_performed']).lower()}`",
            f"- training_performed: `{str(result['training_performed']).lower()}`",
            f"- tiny_overfit_training_only: `{str(result['tiny_overfit_training_only']).lower()}`",
            f"- num_samples: `{result['num_samples']}`",
            f"- policies_trained: `{result['policies_trained']}`",
            f"- train_steps: `{result['train_steps']}`",
            f"- optimizer: `{result['optimizer']}`",
            f"- optimizer_step_performed: `{str(result['optimizer_step_performed']).lower()}`",
            f"- optimizer_step_scope: `{result['optimizer_step_scope']}`",
            f"- all_losses_finite: `{str(result['all_losses_finite']).lower()}`",
            f"- policies_with_loss_decrease: `{result['policies_with_loss_decrease']}`",
            f"- acceptance_pass: `{str(result['acceptance_pass']).lower()}`",
            f"- current_tokens_kept_full: `{str(result['current_tokens_kept_full']).lower()}`",
            f"- train_current_importance: `{str(result['train_current_importance']).lower()}`",
            f"- current_importance_training_performed: `{str(result['current_importance_training_performed']).lower()}`",
            f"- videomae_training_performed: `{str(result['videomae_training_performed']).lower()}`",
            f"- teacher_training_performed: `{str(result['teacher_training_performed']).lower()}`",
            f"- selector_training_performed: `{str(result['selector_training_performed']).lower()}`",
            f"- world_model_large_training_performed: `{str(result['world_model_large_training_performed']).lower()}`",
            f"- token_extraction_performed: `{str(result['token_extraction_performed']).lower()}`",
            f"- importance_generation_performed: `{str(result['importance_generation_performed']).lower()}`",
            f"- download_performed: `{str(result['download_performed']).lower()}`",
            f"- data_token_shards_written: `{str(result['data_token_shards_written']).lower()}`",
            f"- data_importance_shards_written: `{str(result['data_importance_shards_written']).lower()}`",
            f"- checkpoint_saved: `{str(result['checkpoint_saved']).lower()}`",
            f"- tiny_overfit_result_not_final_performance: `{str(result['tiny_overfit_result_not_final_performance']).lower()}`",
            f"- safety_gate_pass: `{str(result['safety_gate_pass']).lower()}`",
            f"- loss_quality_note: `{result['loss_quality_note']}`",
            f"- recommended_step28: `{result['recommended_step28']['name']}`",
            "",
            "## Policy Metrics",
            "",
            "```json",
            json.dumps(result["policy_metric_table"], indent=2, sort_keys=True),
            "```",
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
    print(json.dumps(evaluate_step27_tiny_overfit(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
