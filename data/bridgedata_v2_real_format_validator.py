"""Validate Step21 real tiny BridgeData samples through the Step20 builder."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from data.bridgedata_v2_format_inspector import inspect_bridgedata_v2_subset
from data.bridgedata_v2_manifest_autobuilder import build_manifest_from_directory
from data.bridgedata_v2_manifest_schema import load_bridgedata_manifest_jsonl
from data.bridgedata_v2_window_builder import (
    build_bridgedata_windows_from_manifest,
    summarize_manifest_window_build,
    write_window_manifest_jsonl,
)
from data.long_context_window_spec import window_spec_from_dict


def validate_bridgedata_v2_real_tiny_subset(
    config: dict[str, Any],
    acquisition_summary: dict[str, Any],
) -> dict[str, Any]:
    output_cfg = config["output"]
    validation_path = Path(output_cfg["validation_summary_json"])
    validation_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_dirs = _candidate_subset_dirs(config, acquisition_summary)
    existing_dirs = [path for path in candidate_dirs if path.exists() and path.is_dir()]
    if not existing_dirs:
        summary = _safe_stop_validation(config, acquisition_summary, "No real tiny subset directory exists.")
        _write_validation_outputs(summary, validation_path, Path(output_cfg["validation_summary_md"]))
        return summary

    subset_dir = existing_dirs[0]
    manifest_path = Path(output_cfg["real_manifest_jsonl"])
    autobuild = build_manifest_from_directory(
        subset_dir,
        manifest_path,
        min_frames=int(config["window"]["min_trajectory_len"]),
        max_trajectories=int(config["dataset"]["target_limits"]["max_trajectories"]),
    )
    inspection = inspect_bridgedata_v2_subset(subset_dir)
    if not autobuild["manifest_built"]:
        summary = _safe_stop_validation(config, acquisition_summary, "Could not build a real tiny manifest.")
        summary.update({"real_subset_exists": True, "inspection": inspection, "manifest_autobuild": autobuild})
        _write_validation_outputs(summary, validation_path, Path(output_cfg["validation_summary_md"]))
        return summary

    records = load_bridgedata_manifest_jsonl(manifest_path)
    spec = window_spec_from_dict(config["window"])
    windows = build_bridgedata_windows_from_manifest(records, spec)
    real_window_manifest = Path(output_cfg["real_window_manifest_jsonl"])
    write_window_manifest_jsonl(real_window_manifest, windows)
    window_summary = summarize_manifest_window_build(records, windows, spec)
    frame_counts = [record["num_frames"] for record in records]
    summary = {
        "stage": "bridgedata_v2_real_tiny_format_validation",
        "real_subset_exists": True,
        "real_subset_dir": str(subset_dir),
        "manifest_built": True,
        "manifest_path": str(manifest_path),
        "real_format_validated": bool(records and windows),
        "safe_stop": False,
        "user_provided_subset_required": False,
        "reason": None,
        "num_manifest_records": len(records),
        "num_valid_trajectories": window_summary["num_valid_trajectories"],
        "num_skipped_trajectories": window_summary["num_skipped_trajectories"],
        "skipped_trajectories": window_summary["skipped_trajectories"],
        "frame_count_min": min(frame_counts) if frame_counts else None,
        "frame_count_mean": mean(frame_counts) if frame_counts else None,
        "frame_count_max": max(frame_counts) if frame_counts else None,
        "has_images_likely": inspection.get("has_images_likely"),
        "has_actions_likely": inspection.get("has_actions_likely"),
        "has_language_likely": inspection.get("has_language_likely"),
        "has_goal_image_likely": inspection.get("has_goal_image_likely"),
        "num_windows": len(windows),
        "real_window_manifest_jsonl": str(real_window_manifest),
        "window_spec": {
            "context_len": spec.context_len,
            "current_len": spec.current_len,
            "future_len": spec.future_len,
        },
        "use_action_as_input": False,
        "use_language_as_input": False,
        "use_goal_image_as_input": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "inspection": inspection,
        "manifest_autobuild": autobuild,
    }
    _write_validation_outputs(summary, validation_path, Path(output_cfg["validation_summary_md"]))
    return summary


def _candidate_subset_dirs(config: dict[str, Any], acquisition_summary: dict[str, Any]) -> list[Path]:
    dataset_cfg = config["dataset"]
    candidates = [Path(dataset_cfg["subset_root"])]
    extract_root = acquisition_summary.get("extract_root") or dataset_cfg.get("extract_root")
    if extract_root:
        candidates.append(Path(extract_root))
    download_root = acquisition_summary.get("download_root") or dataset_cfg.get("download_root")
    if download_root:
        candidates.append(Path(download_root))
    return candidates


def _safe_stop_validation(config: dict[str, Any], acquisition_summary: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "stage": "bridgedata_v2_real_tiny_format_validation",
        "real_subset_exists": False,
        "manifest_built": False,
        "real_format_validated": False,
        "safe_stop": True,
        "user_provided_subset_required": True,
        "reason": reason if not acquisition_summary.get("safe_stop") else acquisition_summary.get("reason", reason),
        "num_manifest_records": 0,
        "num_valid_trajectories": 0,
        "num_skipped_trajectories": 0,
        "skipped_trajectories": [],
        "frame_count_min": None,
        "frame_count_mean": None,
        "frame_count_max": None,
        "has_images_likely": None,
        "has_actions_likely": None,
        "has_language_likely": None,
        "has_goal_image_likely": None,
        "num_windows": 0,
        "real_window_manifest_jsonl": config["output"]["real_window_manifest_jsonl"],
        "window_spec": {
            "context_len": int(config["window"]["context_len"]),
            "current_len": int(config["window"]["current_len"]),
            "future_len": int(config["window"]["future_len"]),
        },
        "use_action_as_input": False,
        "use_language_as_input": False,
        "use_goal_image_as_input": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
    }


def _write_validation_outputs(summary: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(_render_validation_md(summary), encoding="utf-8")


def _render_validation_md(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 Real Tiny Validation Summary",
            "",
            f"- real_format_validated: `{str(summary['real_format_validated']).lower()}`",
            f"- safe_stop: `{str(summary['safe_stop']).lower()}`",
            f"- reason: `{summary.get('reason')}`",
            f"- num_manifest_records: `{summary['num_manifest_records']}`",
            f"- num_valid_trajectories: `{summary['num_valid_trajectories']}`",
            f"- num_windows: `{summary['num_windows']}`",
            f"- window spec: `{summary['window_spec']['context_len']}/{summary['window_spec']['current_len']}/{summary['window_spec']['future_len']}`",
            "- training_performed: `false`",
            "- token_extraction_performed: `false`",
            "- importance_generation_performed: `false`",
            "",
        ]
    )
