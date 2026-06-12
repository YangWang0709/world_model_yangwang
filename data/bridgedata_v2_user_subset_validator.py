"""Validate Step22 user-provided BridgeData V2 tiny-subset manifests."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from data.bridgedata_v2_manifest_schema import normalize_bridgedata_manifest_record


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def validate_user_subset_manifest(
    records: list[dict[str, Any]],
    user_subset_root: str | Path,
    *,
    user_subset_exists: bool = True,
    min_frames: int = 24,
    max_images_to_check_exists: int = 100,
    max_images_to_open_optional: int = 20,
    open_images: bool = False,
) -> dict[str, Any]:
    root = Path(user_subset_root)
    if not user_subset_exists or not root.exists() or not root.is_dir():
        return _pending_summary(root, min_frames)

    valid_records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    warnings: list[str] = []
    blocking_errors: list[str] = []
    frame_counts: list[int] = []
    checked_image_count = 0
    opened_image_count = 0
    has_images = False
    has_actions = False
    has_language = False
    has_goal = False
    for raw_record in records:
        try:
            record = normalize_bridgedata_manifest_record(raw_record)
        except Exception as exc:
            blocking_errors.append(f"invalid manifest record: {exc}")
            continue
        trajectory_id = record["trajectory_id"]
        num_frames = int(record["num_frames"])
        frame_counts.append(num_frames)
        frame_paths = _resolve_frame_paths(record, root)
        has_images = has_images or bool(frame_paths)
        has_actions = has_actions or bool(record.get("actions_path") or record.get("actions") is not None)
        has_language = has_language or bool(record.get("language_instruction"))
        has_goal = has_goal or bool(record.get("goal_image_path"))

        skip_reasons: list[str] = []
        if num_frames < min_frames:
            skip_reasons.append(f"num_frames < {min_frames}")
        if len(frame_paths) < min_frames:
            skip_reasons.append(f"resolved frame path count < {min_frames}")
        missing_paths = [str(path) for path in frame_paths[:max_images_to_check_exists] if not path.exists()]
        checked_image_count += min(len(frame_paths), max_images_to_check_exists)
        if missing_paths:
            skip_reasons.append(f"missing checked image paths: {missing_paths[:5]}")
        if open_images and frame_paths:
            opened, open_warnings = _optional_open_images(frame_paths[:max_images_to_open_optional])
            opened_image_count += opened
            warnings.extend(open_warnings)

        if skip_reasons:
            skipped.append({"trajectory_id": trajectory_id, "num_frames": num_frames, "reason": "; ".join(skip_reasons)})
            continue
        valid_records.append(record)

    if records and not valid_records:
        blocking_errors.append("no valid trajectories with existing >=24 frame paths")
    safety_gate_pass = True
    real_format_validated = bool(valid_records and not blocking_errors)
    summary = {
        "stage": "bridgedata_v2_user_subset_ingestion_step22",
        "user_subset_exists": True,
        "pending_user_data": False,
        "real_format_validated": real_format_validated,
        "num_records": len(records),
        "num_valid_trajectories": len(valid_records),
        "num_skipped_trajectories": len(skipped),
        "skipped_trajectories": skipped,
        "valid_trajectory_ids": [record["trajectory_id"] for record in valid_records],
        "frame_count_min": min(frame_counts) if frame_counts else None,
        "frame_count_mean": mean(frame_counts) if frame_counts else None,
        "frame_count_max": max(frame_counts) if frame_counts else None,
        "has_images_likely": has_images if records else None,
        "has_actions_likely": has_actions if records else None,
        "has_language_likely": has_language if records else None,
        "has_goal_image_likely": has_goal if records else None,
        "checked_image_count": checked_image_count,
        "opened_image_count": opened_image_count,
        "warnings": warnings,
        "blocking_errors": blocking_errors,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_image_used_as_input": False,
        "use_action_as_input": False,
        "use_language_as_input": False,
        "use_goal_image_as_input": False,
        "download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "safety_gate_pass": safety_gate_pass,
    }
    return summary


def write_validation_outputs(summary: dict[str, Any], json_path: str | Path, md_path: str | Path) -> None:
    json_output = Path(json_path)
    md_output = Path(md_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_output.write_text(render_validation_summary_md(summary), encoding="utf-8")


def render_validation_summary_md(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 User Subset Validation Summary",
            "",
            f"- user_subset_exists: `{str(summary['user_subset_exists']).lower()}`",
            f"- pending_user_data: `{str(summary['pending_user_data']).lower()}`",
            f"- real_format_validated: `{str(summary['real_format_validated']).lower()}`",
            f"- num_records: `{summary.get('num_records', 0)}`",
            f"- num_valid_trajectories: `{summary['num_valid_trajectories']}`",
            f"- num_skipped_trajectories: `{summary['num_skipped_trajectories']}`",
            f"- frame count min/mean/max: `{summary['frame_count_min']}/{summary['frame_count_mean']}/{summary['frame_count_max']}`",
            f"- blocking_errors: `{len(summary['blocking_errors'])}`",
            "- action_used_as_input: `false`",
            "- language_used_as_input: `false`",
            "- goal_image_used_as_input: `false`",
            "- no download: `true`",
            "- no training: `true`",
            "",
        ]
    )


def _pending_summary(root: Path, min_frames: int) -> dict[str, Any]:
    return {
        "stage": "bridgedata_v2_user_subset_ingestion_step22",
        "user_subset_exists": False,
        "pending_user_data": True,
        "real_format_validated": False,
        "num_records": 0,
        "num_valid_trajectories": 0,
        "num_skipped_trajectories": 0,
        "skipped_trajectories": [],
        "valid_trajectory_ids": [],
        "frame_count_min": None,
        "frame_count_mean": None,
        "frame_count_max": None,
        "has_images_likely": None,
        "has_actions_likely": None,
        "has_language_likely": None,
        "has_goal_image_likely": None,
        "checked_image_count": 0,
        "opened_image_count": 0,
        "warnings": [f"user subset directory is missing: {root}", f"minimum frames per trajectory is {min_frames}"],
        "blocking_errors": [],
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_image_used_as_input": False,
        "use_action_as_input": False,
        "use_language_as_input": False,
        "use_goal_image_as_input": False,
        "download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "safety_gate_pass": True,
    }


def _resolve_frame_paths(record: dict[str, Any], root: Path) -> list[Path]:
    if record.get("frame_paths"):
        return [_resolve_path(root, value) for value in record["frame_paths"]]
    image_dir = record.get("image_dir")
    if image_dir:
        image_root = _resolve_path(root, image_dir)
        if image_root.exists() and image_root.is_dir():
            return sorted(path for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        return [image_root / f"frame_{idx:06d}.jpg" for idx in range(int(record["num_frames"]))]
    return []


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _optional_open_images(paths: list[Path]) -> tuple[int, list[str]]:
    try:
        from PIL import Image
    except Exception:
        return 0, ["PIL is not available; optional image-open validation skipped"]
    opened = 0
    warnings: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            opened += 1
        except Exception as exc:
            warnings.append(f"could not open image {path}: {exc}")
    return opened, warnings
