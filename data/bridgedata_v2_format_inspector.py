"""Lightweight BridgeData V2 tiny-subset layout inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data.bridgedata_v2_manifest_schema import (
    BRIDGEDATA_DATASET_NAME,
    load_bridgedata_manifest_jsonl,
    summarize_bridgedata_manifest_record,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ACTION_FILENAMES = {"actions.npy", "actions.json"}
GOAL_FILENAMES = {"goal.jpg", "goal.jpeg", "goal.png"}


def inspect_bridgedata_v2_subset(local_subset_dir: str | Path | None) -> dict[str, Any]:
    if local_subset_dir is None:
        return _missing_summary(None)
    subset_dir = Path(local_subset_dir)
    if not subset_dir.exists():
        return _missing_summary(subset_dir)
    if not subset_dir.is_dir():
        summary = _missing_summary(subset_dir)
        summary["warnings"] = ["local subset path exists but is not a directory"]
        summary["detected_layout"] = "unknown"
        return summary

    manifest_path = subset_dir / "manifest.jsonl"
    if manifest_path.exists():
        records = load_bridgedata_manifest_jsonl(manifest_path)
        record_summaries = [summarize_bridgedata_manifest_record(record) for record in records]
        return {
            "dataset_name": BRIDGEDATA_DATASET_NAME,
            "local_subset_dir": str(subset_dir),
            "local_subset_exists": True,
            "manifest_found": True,
            "manifest_path": str(manifest_path),
            "detected_layout": "manifest_jsonl",
            "trajectory_count_estimate": len(records),
            "has_images_likely": any(item["has_frame_paths"] or item["has_image_dir"] for item in record_summaries),
            "has_actions_likely": any(item["has_actions_path"] or item["has_inline_actions"] for item in record_summaries),
            "has_language_likely": any(item["has_language_instruction"] for item in record_summaries),
            "has_goal_image_likely": any(item["has_goal_image_path"] for item in record_summaries),
            "record_summaries": record_summaries,
            "detected_files": ["manifest.jsonl"],
            "warnings": [],
            "no_download": True,
        }

    child_dirs = [path for path in sorted(subset_dir.iterdir()) if path.is_dir()]
    detected_files = [path.name for path in sorted(subset_dir.iterdir())[:50]]
    if child_dirs:
        trajectory_summaries = [_inspect_trajectory_dir(path) for path in child_dirs[:1000]]
        return {
            "dataset_name": BRIDGEDATA_DATASET_NAME,
            "local_subset_dir": str(subset_dir),
            "local_subset_exists": True,
            "manifest_found": False,
            "manifest_path": None,
            "detected_layout": "directory_per_trajectory",
            "trajectory_count_estimate": len(child_dirs),
            "has_images_likely": any(item["has_images_likely"] for item in trajectory_summaries),
            "has_actions_likely": any(item["has_actions_likely"] for item in trajectory_summaries),
            "has_language_likely": any(item["has_metadata_json"] for item in trajectory_summaries),
            "has_goal_image_likely": any(item["has_goal_image_likely"] for item in trajectory_summaries),
            "trajectory_summaries": trajectory_summaries,
            "detected_files": detected_files,
            "warnings": ["manifest.jsonl not found; directory-per-trajectory adapter can inspect layout only"],
            "no_download": True,
        }

    return {
        "dataset_name": BRIDGEDATA_DATASET_NAME,
        "local_subset_dir": str(subset_dir),
        "local_subset_exists": True,
        "manifest_found": False,
        "manifest_path": None,
        "detected_layout": "unknown",
        "trajectory_count_estimate": None,
        "has_images_likely": None,
        "has_actions_likely": None,
        "has_language_likely": None,
        "has_goal_image_likely": None,
        "detected_files": detected_files,
        "warnings": ["no manifest.jsonl or trajectory directories detected"],
        "no_download": True,
    }


def find_first_existing_subset_dir(candidates: list[str | Path]) -> Path | None:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_dir():
            return path
    return None


def find_first_existing_manifest(candidates: list[str | Path]) -> Path | None:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path
    return None


def _missing_summary(path: Path | None) -> dict[str, Any]:
    return {
        "dataset_name": BRIDGEDATA_DATASET_NAME,
        "local_subset_dir": str(path) if path is not None else None,
        "local_subset_exists": False,
        "manifest_found": False,
        "manifest_path": None,
        "detected_layout": "missing",
        "trajectory_count_estimate": None,
        "has_images_likely": None,
        "has_actions_likely": None,
        "has_language_likely": None,
        "has_goal_image_likely": None,
        "detected_files": [],
        "warnings": ["local subset directory is missing; fake manifest dry-run is allowed"],
        "no_download": True,
    }


def _inspect_trajectory_dir(path: Path) -> dict[str, Any]:
    names = {item.name for item in path.iterdir()}
    image_dirs = [path / "images", path / "frames"]
    image_count_estimate = 0
    for image_dir in image_dirs:
        if image_dir.exists() and image_dir.is_dir():
            image_count_estimate += sum(1 for child in image_dir.iterdir() if child.suffix.lower() in IMAGE_SUFFIXES)
    return {
        "trajectory_id": path.name,
        "has_metadata_json": "metadata.json" in names,
        "has_images_likely": image_count_estimate > 0 or any(image_dir.exists() for image_dir in image_dirs),
        "image_count_estimate": image_count_estimate,
        "has_actions_likely": bool(names & ACTION_FILENAMES),
        "has_goal_image_likely": bool(names & GOAL_FILENAMES),
        "detected_files": sorted(names)[:50],
    }
