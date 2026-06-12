"""Evaluate Step24 BridgeData TFDS token extraction dry-run outputs."""

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

from data.bridgedata_v2_tfds_token_manifest import read_token_manifest_jsonl, summarize_token_manifest

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_token_extraction_step24.yaml"
DOC_REPORT = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TFDS_TOKEN_EXTRACTION_REPORT.md"


def evaluate_step24_token_extraction(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    output = config["output"]
    clip_summary_path = Path(output["clip_export_summary_json"])
    token_summary_path = Path(output["token_summary_json"])
    token_manifest_path = Path(output["token_manifest_jsonl"])
    eval_json_path = Path(output["eval_json"])
    eval_md_path = Path(output["eval_md"])

    clip_summary = _read_json_or_empty(clip_summary_path)
    token_summary = _read_json_or_empty(token_summary_path)
    manifest_records = read_token_manifest_jsonl(token_manifest_path) if token_manifest_path.exists() else []
    manifest_summary = summarize_token_manifest(manifest_records)

    token_success = bool(token_summary.get("token_extraction_performed")) and manifest_summary["num_windows_tokenized"] > 0
    safe_stop = bool(clip_summary.get("safe_stop")) or bool(token_summary.get("safe_stop"))
    model_missing = bool(token_summary.get("model_missing"))
    passed = bool(clip_summary.get("clip_export_performed")) and token_success and not safe_stop
    recommended = (
        {
            "name": "BridgeData V2 TFDS mini-shard predictive importance dry-run",
            "condition": "token extraction dry-run succeeded",
            "no_training": True,
        }
        if passed
        else {
            "name": "Fix local VideoMAE cache / restore local model before token extraction",
            "condition": "token extraction dry-run did not succeed",
            "no_training": True,
        }
    )
    result = {
        "stage": "bridgedata_v2_tfds_token_extraction_step24",
        "pass": passed,
        "safe_stop": safe_stop,
        "model_missing": model_missing,
        "clip_export_performed": bool(clip_summary.get("clip_export_performed")),
        "token_extraction_performed": bool(token_summary.get("token_extraction_performed")),
        "num_windows_exported": int(clip_summary.get("num_windows_exported") or 0),
        "num_windows_tokenized": manifest_summary["num_windows_tokenized"],
        "num_token_artifacts": manifest_summary["num_token_artifacts"],
        "image_field": token_summary.get("image_field") or clip_summary.get("image_field"),
        "image_field_is_metadata_flag": bool(
            token_summary.get("image_field_is_metadata_flag") or clip_summary.get("image_field_is_metadata_flag")
        ),
        "context_clip_shape_example": clip_summary.get("context_shape"),
        "current_clip_shape_example": clip_summary.get("current_shape"),
        "future_clip_shape_example": clip_summary.get("future_shape"),
        "context_token_shape_example": manifest_summary["context_token_shape_example"],
        "current_token_shape_example": manifest_summary["current_token_shape_example"],
        "future_token_shape_example": manifest_summary["future_token_shape_example"],
        "model_loaded_local_only": bool(token_summary.get("model_loaded_local_only")),
        "model_download_performed": bool(token_summary.get("model_download_performed")),
        "training_performed": bool(token_summary.get("training_performed")),
        "importance_generation_performed": bool(token_summary.get("importance_generation_performed")),
        "large_token_shards_generated": bool(token_summary.get("large_token_shards_generated")),
        "data_token_shards_written": bool(token_summary.get("data_token_shards_written")),
        "safety_gate_pass": _safety_gate_pass(clip_summary, token_summary),
        "recommended_step25": recommended,
    }
    eval_json_path.parent.mkdir(parents=True, exist_ok=True)
    eval_json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    report = _format_report(result, clip_summary, token_summary)
    eval_md_path.parent.mkdir(parents=True, exist_ok=True)
    eval_md_path.write_text(report, encoding="utf-8")
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.write_text(report, encoding="utf-8")
    return result


def _safety_gate_pass(clip_summary: dict[str, Any], token_summary: dict[str, Any]) -> bool:
    return (
        bool(clip_summary.get("safety_gate_pass", True))
        and bool(token_summary.get("safety_gate_pass", True))
        and not bool(clip_summary.get("download_performed"))
        and not bool(token_summary.get("model_download_performed"))
        and not bool(token_summary.get("training_performed"))
        and not bool(token_summary.get("importance_generation_performed"))
        and not bool(token_summary.get("large_token_shards_generated"))
        and not bool(token_summary.get("data_token_shards_written"))
    )


def _format_report(result: dict[str, Any], clip_summary: dict[str, Any], token_summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 TFDS Token Extraction Report",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- safe_stop: `{str(result['safe_stop']).lower()}`",
            f"- model_missing: `{str(result['model_missing']).lower()}`",
            f"- image_field: `{result['image_field']}`",
            f"- image_field_is_metadata_flag: `{str(result['image_field_is_metadata_flag']).lower()}`",
            f"- clip_export_performed: `{str(result['clip_export_performed']).lower()}`",
            f"- token_extraction_performed: `{str(result['token_extraction_performed']).lower()}`",
            f"- num_windows_exported: `{result['num_windows_exported']}`",
            f"- num_windows_tokenized: `{result['num_windows_tokenized']}`",
            f"- context_clip_shape_example: `{result['context_clip_shape_example']}`",
            f"- current_clip_shape_example: `{result['current_clip_shape_example']}`",
            f"- future_clip_shape_example: `{result['future_clip_shape_example']}`",
            f"- context_token_shape_example: `{result['context_token_shape_example']}`",
            f"- current_token_shape_example: `{result['current_token_shape_example']}`",
            f"- future_token_shape_example: `{result['future_token_shape_example']}`",
            f"- model_loaded_local_only: `{str(result['model_loaded_local_only']).lower()}`",
            f"- model_download_performed: `{str(result['model_download_performed']).lower()}`",
            f"- training_performed: `{str(result['training_performed']).lower()}`",
            f"- importance_generation_performed: `{str(result['importance_generation_performed']).lower()}`",
            f"- large_token_shards_generated: `{str(result['large_token_shards_generated']).lower()}`",
            f"- data_token_shards_written: `{str(result['data_token_shards_written']).lower()}`",
            f"- safety_gate_pass: `{str(result['safety_gate_pass']).lower()}`",
            f"- recommended_step25: `{result['recommended_step25']['name']}`",
            "",
            "## Clip Export Summary",
            "",
            "```json",
            json.dumps(clip_summary, indent=2, sort_keys=True),
            "```",
            "",
            "## Token Summary",
            "",
            "```json",
            json.dumps(token_summary, indent=2, sort_keys=True),
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
    print(json.dumps(evaluate_step24_token_extraction(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
