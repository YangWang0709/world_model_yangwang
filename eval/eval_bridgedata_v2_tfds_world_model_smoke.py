"""Evaluate Step26 BridgeData TFDS world-model smoke outputs."""

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

from data.bridgedata_v2_tfds_world_model_smoke_metrics import LOSS_QUALITY_NOTE

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_world_model_smoke_step26.yaml"
DOC_REPORT = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TFDS_WORLD_MODEL_SMOKE_REPORT.md"


def evaluate_step26_world_model_smoke(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved_config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    output = config["output"]
    batch_summary = _read_json_or_empty(output["batch_summary_json"])
    policy_payload = _read_json_or_empty(output["policy_metrics_json"])
    forward_summary = _read_json_or_empty(output["forward_loss_json"])
    smoke_summary = _read_json_or_empty(output["smoke_summary_json"])
    policy_metrics = policy_payload.get("policy_metrics", [])
    safe_stop = bool(smoke_summary.get("safe_stop", not smoke_summary))
    safety_gate_pass = _safety_gate_pass(smoke_summary)
    passed = _pass(smoke_summary, safe_stop, safety_gate_pass)
    result = {
        "stage": config["stage"],
        "pass": passed,
        "safe_stop": safe_stop,
        "world_model_smoke_performed": bool(smoke_summary.get("world_model_smoke_performed")),
        "num_samples": int(smoke_summary.get("num_samples") or 0),
        "policies_evaluated": smoke_summary.get("policies_evaluated") or [],
        "topk_values": smoke_summary.get("topk_values") or [],
        "all_losses_finite": bool(smoke_summary.get("all_losses_finite")),
        "current_tokens_kept_full": bool(smoke_summary.get("current_tokens_kept_full")),
        "train_current_importance": bool(smoke_summary.get("train_current_importance")),
        "optimizer_step_performed": bool(smoke_summary.get("optimizer_step_performed")),
        "training_performed": bool(smoke_summary.get("training_performed")),
        "teacher_training_performed": bool(smoke_summary.get("teacher_training_performed")),
        "selector_training_performed": bool(smoke_summary.get("selector_training_performed")),
        "world_model_training_performed": bool(smoke_summary.get("world_model_training_performed")),
        "token_extraction_performed": bool(smoke_summary.get("token_extraction_performed")),
        "importance_generation_performed": bool(smoke_summary.get("importance_generation_performed")),
        "download_performed": bool(smoke_summary.get("download_performed")),
        "model_download_performed": bool(smoke_summary.get("model_download_performed")),
        "data_token_shards_written": bool(smoke_summary.get("data_token_shards_written")),
        "data_importance_shards_written": bool(smoke_summary.get("data_importance_shards_written")),
        "random_init_result_not_scientific": bool(smoke_summary.get("random_init_result_not_scientific")),
        "policy_metrics": policy_metrics,
        "policy_metric_table": smoke_summary.get("policy_metric_table") or [],
        "batch_summary": batch_summary,
        "forward_loss_summary": forward_summary,
        "loss_quality_note": LOSS_QUALITY_NOTE,
        "reason": smoke_summary.get("reason"),
        "safety_gate_pass": safety_gate_pass,
        "recommended_step27": {
            "name": "Step27 BridgeData V2 TFDS mini-shard tiny overfit world-model training",
            "condition": "forward smoke passed",
            "scope": "very small, explicit training allowed only in Step27",
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


def _pass(summary: dict[str, Any], safe_stop: bool, safety_gate_pass: bool) -> bool:
    return (
        not safe_stop
        and safety_gate_pass
        and bool(summary.get("world_model_smoke_performed"))
        and int(summary.get("num_samples") or 0) >= 1
        and bool(summary.get("all_losses_finite"))
        and bool(summary.get("current_tokens_kept_full"))
        and not bool(summary.get("train_current_importance"))
        and not bool(summary.get("optimizer_step_performed"))
        and not bool(summary.get("training_performed"))
        and not bool(summary.get("teacher_training_performed"))
        and not bool(summary.get("selector_training_performed"))
        and not bool(summary.get("world_model_training_performed"))
        and not bool(summary.get("token_extraction_performed"))
        and not bool(summary.get("importance_generation_performed"))
        and not bool(summary.get("download_performed"))
        and not bool(summary.get("model_download_performed"))
        and not bool(summary.get("data_token_shards_written"))
        and not bool(summary.get("data_importance_shards_written"))
    )


def _safety_gate_pass(summary: dict[str, Any]) -> bool:
    return (
        bool(summary.get("safety_gate_pass", True))
        and not bool(summary.get("download_performed"))
        and not bool(summary.get("model_download_performed"))
        and not bool(summary.get("training_performed"))
        and not bool(summary.get("optimizer_step_performed"))
        and not bool(summary.get("teacher_training_performed"))
        and not bool(summary.get("selector_training_performed"))
        and not bool(summary.get("world_model_training_performed"))
        and not bool(summary.get("token_extraction_performed"))
        and not bool(summary.get("importance_generation_performed"))
        and not bool(summary.get("data_token_shards_written"))
        and not bool(summary.get("data_importance_shards_written"))
    )


def _format_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 TFDS World-Model Smoke Report",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- safe_stop: `{str(result['safe_stop']).lower()}`",
            f"- world_model_smoke_performed: `{str(result['world_model_smoke_performed']).lower()}`",
            f"- num_samples: `{result['num_samples']}`",
            f"- policies_evaluated: `{result['policies_evaluated']}`",
            f"- topk_values: `{result['topk_values']}`",
            f"- all_losses_finite: `{str(result['all_losses_finite']).lower()}`",
            f"- current_tokens_kept_full: `{str(result['current_tokens_kept_full']).lower()}`",
            f"- train_current_importance: `{str(result['train_current_importance']).lower()}`",
            f"- optimizer_step_performed: `{str(result['optimizer_step_performed']).lower()}`",
            f"- training_performed: `{str(result['training_performed']).lower()}`",
            f"- token_extraction_performed: `{str(result['token_extraction_performed']).lower()}`",
            f"- importance_generation_performed: `{str(result['importance_generation_performed']).lower()}`",
            f"- download_performed: `{str(result['download_performed']).lower()}`",
            f"- data_token_shards_written: `{str(result['data_token_shards_written']).lower()}`",
            f"- data_importance_shards_written: `{str(result['data_importance_shards_written']).lower()}`",
            f"- random_init_result_not_scientific: `{str(result['random_init_result_not_scientific']).lower()}`",
            f"- safety_gate_pass: `{str(result['safety_gate_pass']).lower()}`",
            f"- loss_quality_note: `{result['loss_quality_note']}`",
            f"- recommended_step27: `{result['recommended_step27']['name']}`",
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
    print(json.dumps(evaluate_step26_world_model_smoke(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
