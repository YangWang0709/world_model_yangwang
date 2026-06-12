"""Rebuild Step23 TFDS mini manifest/windows with resolved RLDS field policy."""

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

from data.bridgedata_v2_rlds_field_resolver import resolve_rlds_fields
from data.bridgedata_v2_rlds_to_manifest import schema_summary_to_manifest_records, write_manifest_and_summary
from data.bridgedata_v2_window_builder import (
    build_bridgedata_windows_from_manifest,
    summarize_manifest_window_build,
    write_window_manifest_jsonl,
)
from data.long_context_window_spec import window_spec_from_dict

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_field_resolver_step23_5.yaml"


def rebuild_resolved_manifest_and_windows(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    schema_path = Path(config["input"]["step23_schema_summary"])
    resolved_path = Path(config["output"]["resolved_fields_json"])
    manifest_path = Path(config["output"]["resolved_manifest_jsonl"])
    manifest_summary_path = Path(config["output"]["resolved_manifest_summary_json"])
    window_path = Path(config["output"]["resolved_window_manifest_jsonl"])
    window_summary_path = Path(config["output"]["resolved_window_summary_json"])
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    if not schema_path.exists():
        resolved = _safe_stop_resolved("Step23 schema summary is missing.")
        resolved_path.write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
        manifest_summary = _write_empty_manifest_summary(manifest_path, manifest_summary_path, resolved)
        window_summary = _write_empty_window_summary(window_path, window_summary_path, config, resolved)
        return {"resolved_fields": resolved, "manifest_summary": manifest_summary, "window_summary": window_summary}

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    resolved_core = resolve_rlds_fields(schema.get("candidate_fields", {}), config.get("field_policy"))
    resolved = {
        "stage": "bridgedata_v2_tfds_field_resolver_step23_5",
        "safe_stop": False,
        "reason": None,
        **resolved_core,
        "download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
    }
    resolved_path.write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    schema["resolved_fields"] = resolved_core
    records = schema_summary_to_manifest_records(
        schema,
        min_frames=int(config["window"]["min_trajectory_len"]),
        max_valid_trajectories=int(config["sample_limits"]["max_valid_trajectories"]),
        field_policy=config.get("field_policy"),
    )
    manifest_summary = write_manifest_and_summary(
        records,
        manifest_path,
        manifest_summary_path,
        schema_summary=schema,
        min_valid_trajectories=int(config["sample_limits"]["min_valid_trajectories"]),
    )
    spec = window_spec_from_dict(config["window"])
    windows = []
    if records and not manifest_summary.get("safe_stop"):
        windows = build_bridgedata_windows_from_manifest(records, spec, max_windows=int(config["sample_limits"]["max_windows"]))
        write_window_manifest_jsonl(window_path, windows)
    elif window_path.exists():
        window_path.unlink()
    build_summary = summarize_manifest_window_build(records, windows, spec) if records else _empty_window_build_summary(spec)
    window_summary = {
        "stage": "bridgedata_v2_tfds_field_resolver_window_builder",
        "pass": bool(resolved["image_field_valid"] and windows),
        "safe_stop": not bool(resolved["image_field_valid"] and windows),
        "reason": None if resolved["image_field_valid"] and windows else "resolved image field or window generation failed",
        "resolved_fields": resolved,
        "num_manifest_records": len(records),
        "num_valid_trajectories": build_summary["num_valid_trajectories"],
        "num_windows": len(windows),
        "context_len": spec.context_len,
        "current_len": spec.current_len,
        "future_len": spec.future_len,
        "resolved_manifest_path": str(manifest_path),
        "resolved_window_manifest_path": str(window_path),
        "resolved_window_manifest_exists": window_path.exists(),
        "download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "window_summary": build_summary,
    }
    window_summary_path.parent.mkdir(parents=True, exist_ok=True)
    window_summary_path.write_text(json.dumps(window_summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"resolved_fields": resolved, "manifest_summary": manifest_summary, "window_summary": window_summary}


def _safe_stop_resolved(reason: str) -> dict[str, Any]:
    return {
        "stage": "bridgedata_v2_tfds_field_resolver_step23_5",
        "safe_stop": True,
        "reason": reason,
        "image_field": None,
        "image_field_valid": False,
        "image_field_is_metadata_flag": False,
        "image_rejected_fields": [],
        "action_field": None,
        "action_used_as_input": False,
        "language_field": None,
        "language_used_as_input": False,
        "goal_field": None,
        "goal_used_as_input": False,
        "blocking_errors": [reason],
        "warnings": [],
        "download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
    }


def _write_empty_manifest_summary(manifest_path: Path, summary_path: Path, resolved: dict[str, Any]) -> dict[str, Any]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("", encoding="utf-8")
    summary = {
        "stage": "bridgedata_v2_tfds_mini_manifest",
        "real_tfds_validated": False,
        "safe_stop": True,
        "reason": resolved["reason"],
        "manifest_path": str(manifest_path),
        "num_manifest_records": 0,
        "num_valid_trajectories": 0,
        "num_skipped_trajectories": 0,
        "resolved_fields": resolved,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "save_video_tensors": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _write_empty_window_summary(window_path: Path, summary_path: Path, config: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
    if window_path.exists():
        window_path.unlink()
    spec = window_spec_from_dict(config["window"])
    summary = {
        "stage": "bridgedata_v2_tfds_field_resolver_window_builder",
        "pass": False,
        "safe_stop": True,
        "reason": resolved["reason"],
        "resolved_fields": resolved,
        "num_manifest_records": 0,
        "num_valid_trajectories": 0,
        "num_windows": 0,
        "context_len": spec.context_len,
        "current_len": spec.current_len,
        "future_len": spec.future_len,
        "resolved_window_manifest_path": str(window_path),
        "resolved_window_manifest_exists": False,
        "download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _empty_window_build_summary(spec) -> dict[str, Any]:
    return {
        "num_trajectories": 0,
        "num_valid_trajectories": 0,
        "num_skipped_trajectories": 0,
        "skipped_trajectories": [],
        "num_windows": 0,
        "context_len": spec.context_len,
        "current_len": spec.current_len,
        "future_len": spec.future_len,
        "required_total_frames": spec.required_total_frames,
        "window_summary": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(rebuild_resolved_manifest_and_windows(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
