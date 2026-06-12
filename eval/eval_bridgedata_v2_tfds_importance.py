"""Evaluate Step25 BridgeData TFDS importance dry-run outputs."""

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

from data.bridgedata_v2_tfds_importance_manifest import read_importance_manifest_jsonl, summarize_importance_manifest

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_importance_step25.yaml"
DOC_REPORT = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TFDS_IMPORTANCE_REPORT.md"
LABEL_QUALITY_NOTE = "proxy dry-run only; not final teacher label"


def evaluate_step25_importance(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved_config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    output = config["output"]
    summary_path = Path(output["importance_summary_json"])
    manifest_path = Path(output["importance_manifest_jsonl"])
    eval_json_path = Path(output["eval_json"])
    eval_md_path = Path(output["eval_md"])

    summary = _read_json_or_empty(summary_path)
    records = read_importance_manifest_jsonl(manifest_path) if manifest_path.exists() else []
    manifest_summary = summarize_importance_manifest(records)
    safe_stop = bool(summary.get("safe_stop", not summary))
    safety_gate_pass = _safety_gate_pass(summary)
    passed = _pass(summary, manifest_summary, safe_stop, safety_gate_pass)
    result = {
        "stage": config["stage"],
        "pass": passed,
        "safe_stop": safe_stop,
        "importance_generation_performed": bool(summary.get("importance_generation_performed")),
        "method": summary.get("method"),
        "num_samples": int(summary.get("num_samples") or 0),
        "num_importance_artifacts": int(summary.get("num_importance_artifacts") or 0),
        "context_importance_shape_example": summary.get("context_importance_shape_example"),
        "temporal_importance_shape_example": summary.get("temporal_importance_shape_example"),
        "spatial_importance_shape_example": summary.get("spatial_importance_shape_example"),
        "importance_norm_min": summary.get("importance_norm_min"),
        "importance_norm_max": summary.get("importance_norm_max"),
        "current_tokens_kept_full": bool(summary.get("current_tokens_kept_full")),
        "train_current_importance": bool(summary.get("train_current_importance")),
        "download_performed": bool(summary.get("download_performed")),
        "model_download_performed": bool(summary.get("model_download_performed")),
        "training_performed": bool(summary.get("training_performed")),
        "teacher_training_performed": bool(summary.get("teacher_training_performed")),
        "selector_training_performed": bool(summary.get("selector_training_performed")),
        "world_model_training_performed": bool(summary.get("world_model_training_performed")),
        "token_extraction_performed": bool(summary.get("token_extraction_performed")),
        "large_importance_shards_generated": bool(summary.get("large_importance_shards_generated")),
        "data_importance_shards_written": bool(summary.get("data_importance_shards_written")),
        "data_token_shards_written": bool(summary.get("data_token_shards_written")),
        "label_quality_note": LABEL_QUALITY_NOTE,
        "reason": summary.get("reason"),
        "safety_gate_pass": safety_gate_pass,
        "recommended_step26": {
            "name": "BridgeData V2 TFDS mini-shard context bottleneck world-model smoke",
            "condition": "importance dry-run succeeded",
            "no_training_or_tiny_training": "decide next",
        },
        "manifest_summary": manifest_summary,
    }
    eval_json_path.parent.mkdir(parents=True, exist_ok=True)
    eval_json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    report = _format_report(result, summary)
    eval_md_path.parent.mkdir(parents=True, exist_ok=True)
    eval_md_path.write_text(report, encoding="utf-8")
    if resolved_config_path == DEFAULT_CONFIG.resolve():
        DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
        DOC_REPORT.write_text(report, encoding="utf-8")
    return result


def _pass(
    summary: dict[str, Any],
    manifest_summary: dict[str, Any],
    safe_stop: bool,
    safety_gate_pass: bool,
) -> bool:
    return (
        not safe_stop
        and safety_gate_pass
        and bool(summary.get("importance_generation_performed"))
        and int(summary.get("num_samples") or 0) >= 1
        and int(manifest_summary.get("num_samples") or 0) >= 1
        and summary.get("context_importance_shape_example") == [16, 392]
        and summary.get("temporal_importance_shape_example") == [16]
        and summary.get("spatial_importance_shape_example") == [392]
        and float(summary.get("importance_norm_min")) >= 0.0
        and float(summary.get("importance_norm_max")) <= 1.0
        and bool(summary.get("current_tokens_kept_full"))
        and not bool(summary.get("train_current_importance"))
        and not bool(summary.get("download_performed"))
        and not bool(summary.get("model_download_performed"))
        and not bool(summary.get("training_performed"))
        and not bool(summary.get("teacher_training_performed"))
        and not bool(summary.get("selector_training_performed"))
        and not bool(summary.get("world_model_training_performed"))
        and not bool(summary.get("token_extraction_performed"))
        and not bool(summary.get("large_importance_shards_generated"))
        and not bool(summary.get("data_importance_shards_written"))
    )


def _safety_gate_pass(summary: dict[str, Any]) -> bool:
    return (
        bool(summary.get("safety_gate_pass", True))
        and not bool(summary.get("download_performed"))
        and not bool(summary.get("model_download_performed"))
        and not bool(summary.get("training_performed"))
        and not bool(summary.get("teacher_training_performed"))
        and not bool(summary.get("selector_training_performed"))
        and not bool(summary.get("world_model_training_performed"))
        and not bool(summary.get("token_extraction_performed"))
        and not bool(summary.get("large_importance_shards_generated"))
        and not bool(summary.get("data_importance_shards_written"))
        and not bool(summary.get("data_token_shards_written"))
    )


def _format_report(result: dict[str, Any], summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 TFDS Importance Report",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- safe_stop: `{str(result['safe_stop']).lower()}`",
            f"- importance_generation_performed: `{str(result['importance_generation_performed']).lower()}`",
            f"- method: `{result['method']}`",
            f"- num_samples: `{result['num_samples']}`",
            f"- context_importance_shape_example: `{result['context_importance_shape_example']}`",
            f"- temporal_importance_shape_example: `{result['temporal_importance_shape_example']}`",
            f"- spatial_importance_shape_example: `{result['spatial_importance_shape_example']}`",
            f"- importance_norm_min: `{result['importance_norm_min']}`",
            f"- importance_norm_max: `{result['importance_norm_max']}`",
            f"- current_tokens_kept_full: `{str(result['current_tokens_kept_full']).lower()}`",
            f"- train_current_importance: `{str(result['train_current_importance']).lower()}`",
            f"- download_performed: `{str(result['download_performed']).lower()}`",
            f"- training_performed: `{str(result['training_performed']).lower()}`",
            f"- token_extraction_performed: `{str(result['token_extraction_performed']).lower()}`",
            f"- large_importance_shards_generated: `{str(result['large_importance_shards_generated']).lower()}`",
            f"- data_importance_shards_written: `{str(result['data_importance_shards_written']).lower()}`",
            f"- safety_gate_pass: `{str(result['safety_gate_pass']).lower()}`",
            f"- label_quality_note: `{result['label_quality_note']}`",
            f"- recommended_step26: `{result['recommended_step26']['name']}`",
            "",
            "## Summary JSON",
            "",
            "```json",
            json.dumps(summary, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    print(json.dumps(evaluate_step25_importance(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
