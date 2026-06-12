"""Manifest helpers for user-provided BridgeData V2 tiny subsets.

The schema is an internal stable entry point for Step20. It is not a claim
about the exact upstream BridgeData V2 storage format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BRIDGEDATA_DATASET_NAME = "BridgeData V2"
REQUIRED_FIELDS = {"trajectory_id", "num_frames"}
OPTIONAL_FIELDS = {
    "dataset_name",
    "split",
    "frame_paths",
    "image_dir",
    "camera_names",
    "actions_path",
    "actions",
    "language_instruction",
    "goal_image_path",
    "task_id",
    "environment_id",
    "metadata",
}


def validate_bridgedata_manifest_record(record: dict[str, Any], strict: bool = False) -> bool:
    if not isinstance(record, dict):
        raise ValueError("manifest record must be a dict")
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        raise ValueError(f"missing required manifest fields: {missing}")
    if not isinstance(record.get("trajectory_id"), str) or not record["trajectory_id"]:
        raise ValueError("trajectory_id must be a non-empty string")
    if not isinstance(record.get("num_frames"), int) or record["num_frames"] <= 0:
        raise ValueError("num_frames must be a positive integer")

    dataset_name = record.get("dataset_name", BRIDGEDATA_DATASET_NAME)
    if dataset_name != BRIDGEDATA_DATASET_NAME:
        raise ValueError(f"dataset_name must be {BRIDGEDATA_DATASET_NAME!r}")

    frame_paths = record.get("frame_paths")
    if frame_paths is not None:
        if not isinstance(frame_paths, list) or not all(isinstance(path, str) for path in frame_paths):
            raise ValueError("frame_paths must be list[str] or None")
        if strict and len(frame_paths) != record["num_frames"]:
            raise ValueError("strict=True requires len(frame_paths) == num_frames")

    camera_names = record.get("camera_names")
    if camera_names is not None:
        if not isinstance(camera_names, list) or not all(isinstance(name, str) for name in camera_names):
            raise ValueError("camera_names must be list[str] or None")

    language = record.get("language_instruction")
    if language is not None and not isinstance(language, str):
        raise ValueError("language_instruction must be str or None")

    metadata = record.get("metadata", {})
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be dict or None")

    for key in ["image_dir", "actions_path", "goal_image_path", "task_id", "environment_id", "split"]:
        value = record.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} must be str or None")

    return True


def normalize_bridgedata_manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    validate_bridgedata_manifest_record(record, strict=False)
    normalized = {
        "dataset_name": record.get("dataset_name", BRIDGEDATA_DATASET_NAME),
        "split": record.get("split", "train"),
        "trajectory_id": record["trajectory_id"],
        "num_frames": record["num_frames"],
        "frame_paths": record.get("frame_paths"),
        "image_dir": record.get("image_dir"),
        "camera_names": record.get("camera_names") or [],
        "actions_path": record.get("actions_path"),
        "actions": record.get("actions"),
        "language_instruction": record.get("language_instruction"),
        "goal_image_path": record.get("goal_image_path"),
        "task_id": record.get("task_id"),
        "environment_id": record.get("environment_id"),
        "metadata": record.get("metadata") or {},
    }
    validate_bridgedata_manifest_record(normalized, strict=False)
    return normalized


def summarize_bridgedata_manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_bridgedata_manifest_record(record)
    return {
        "dataset_name": normalized["dataset_name"],
        "split": normalized["split"],
        "trajectory_id": normalized["trajectory_id"],
        "num_frames": normalized["num_frames"],
        "has_frame_paths": bool(normalized.get("frame_paths")),
        "frame_path_count": len(normalized.get("frame_paths") or []),
        "has_image_dir": normalized.get("image_dir") is not None,
        "camera_count": len(normalized.get("camera_names") or []),
        "has_actions_path": normalized.get("actions_path") is not None,
        "has_inline_actions": normalized.get("actions") is not None,
        "has_language_instruction": normalized.get("language_instruction") is not None,
        "has_goal_image_path": normalized.get("goal_image_path") is not None,
        "has_task_id": normalized.get("task_id") is not None,
        "has_environment_id": normalized.get("environment_id") is not None,
        "metadata_keys": sorted((normalized.get("metadata") or {}).keys()),
    }


def load_bridgedata_manifest_jsonl(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            records.append(normalize_bridgedata_manifest_record(record))
    return records


def write_bridgedata_manifest_jsonl(path: str | Path, records: list[dict[str, Any]]) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            normalized = normalize_bridgedata_manifest_record(record)
            handle.write(json.dumps(normalized, sort_keys=True) + "\n")
    return manifest_path


def make_fake_bridgedata_manifest_records(
    num_trajectories: int = 3,
    num_frames: int = 40,
) -> list[dict[str, Any]]:
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    records = []
    for idx in range(num_trajectories):
        trajectory_id = f"bridge_fake_traj_{idx:06d}"
        frame_paths = [f"{trajectory_id}/images/frame_{frame_idx:06d}.jpg" for frame_idx in range(num_frames)]
        records.append(
            normalize_bridgedata_manifest_record(
                {
                    "dataset_name": BRIDGEDATA_DATASET_NAME,
                    "split": "dryrun",
                    "trajectory_id": trajectory_id,
                    "num_frames": num_frames,
                    "frame_paths": frame_paths,
                    "image_dir": f"{trajectory_id}/images",
                    "camera_names": ["main"],
                    "actions_path": f"{trajectory_id}/actions.npy",
                    "actions": None,
                    "language_instruction": "fake instruction, not encoded",
                    "goal_image_path": f"{trajectory_id}/goal.jpg",
                    "task_id": "fake_task",
                    "environment_id": "fake_env",
                    "metadata": {"source": "fake_manifest_step20", "no_real_data": True},
                }
            )
        )
    return records
