"""Manifest helpers for Step22 user-provided BridgeData V2 tiny subsets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.bridgedata_v2_manifest_schema import (
    load_bridgedata_manifest_jsonl,
    normalize_bridgedata_manifest_record,
    write_bridgedata_manifest_jsonl,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def build_or_normalize_user_subset_manifest(
    user_subset_root: str | Path,
    output_manifest: str | Path,
    *,
    manifest_path: str | Path | None = None,
    min_frames: int = 24,
    max_trajectories: int = 10,
) -> dict[str, Any]:
    root = Path(user_subset_root)
    output_path = Path(output_manifest)
    source_manifest = Path(manifest_path) if manifest_path is not None else root / "manifest.jsonl"
    if not root.exists() or not root.is_dir():
        return _missing_summary(root, output_path)

    if source_manifest.exists() and source_manifest.is_file():
        records = load_bridgedata_manifest_jsonl(source_manifest)
        write_bridgedata_manifest_jsonl(output_path, records)
        return {
            "stage": "bridgedata_v2_user_subset_ingestion_step22",
            "pending_user_data": False,
            "user_subset_exists": True,
            "manifest_exists": True,
            "generated_manifest_exists": output_path.exists(),
            "manifest_source": "existing_manifest_jsonl",
            "manifest_path": str(output_path),
            "num_manifest_records": len(records),
            "num_valid_trajectories": sum(1 for record in records if int(record.get("num_frames", 0)) >= min_frames),
            "num_skipped_trajectories": sum(1 for record in records if int(record.get("num_frames", 0)) < min_frames),
            "skipped_trajectories": [
                {
                    "trajectory_id": record.get("trajectory_id"),
                    "num_frames": record.get("num_frames"),
                    "reason": f"num_frames < {min_frames}",
                }
                for record in records
                if int(record.get("num_frames", 0)) < min_frames
            ],
            "warnings": [],
            "download_performed": False,
        }

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    trajectory_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()][:max_trajectories]
    for traj_dir in trajectory_dirs:
        result = _record_from_trajectory_dir(root, traj_dir, min_frames)
        if result["record"] is not None:
            records.append(result["record"])
        else:
            skipped.append(result["skip"])
    write_bridgedata_manifest_jsonl(output_path, records)
    warnings = [] if trajectory_dirs else ["no trajectory directories found"]
    return {
        "stage": "bridgedata_v2_user_subset_ingestion_step22",
        "pending_user_data": False,
        "user_subset_exists": True,
        "manifest_exists": False,
        "generated_manifest_exists": output_path.exists(),
        "manifest_source": "directory_per_trajectory",
        "manifest_path": str(output_path),
        "num_manifest_records": len(records),
        "num_valid_trajectories": len(records),
        "num_skipped_trajectories": len(skipped),
        "skipped_trajectories": skipped,
        "warnings": warnings,
        "download_performed": False,
    }


def write_manifest_summary(summary: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _record_from_trajectory_dir(root: Path, traj_dir: Path, min_frames: int) -> dict[str, Any]:
    image_dir = _find_image_dir(traj_dir)
    frame_paths: list[str] = []
    if image_dir is not None:
        frame_paths = sorted(
            str(path.relative_to(root)).replace("\\", "/")
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if len(frame_paths) < min_frames:
        return {
            "record": None,
            "skip": {
                "trajectory_id": traj_dir.name,
                "num_frames": len(frame_paths),
                "reason": f"num_frames < {min_frames}",
            },
        }
    metadata = _read_metadata(traj_dir / "metadata.json")
    record = normalize_bridgedata_manifest_record(
        {
            "dataset_name": "BridgeData V2",
            "split": metadata.get("split", "user_subset"),
            "trajectory_id": metadata.get("trajectory_id", traj_dir.name),
            "num_frames": len(frame_paths),
            "frame_paths": frame_paths,
            "image_dir": str(image_dir.relative_to(root)).replace("\\", "/") if image_dir is not None else None,
            "camera_names": metadata.get("camera_names") or ["main"],
            "actions_path": metadata.get("actions_path") or _first_existing_relative(root, traj_dir, ["actions.npy", "actions.json"]),
            "language_instruction": metadata.get("language_instruction"),
            "goal_image_path": metadata.get("goal_image_path") or _first_existing_relative(root, traj_dir, ["goal.jpg", "goal.jpeg", "goal.png"]),
            "task_id": metadata.get("task_id"),
            "environment_id": metadata.get("environment_id"),
            "metadata": {"source": "user_provided_tiny_subset", **(metadata.get("metadata") or {})},
        }
    )
    return {"record": record, "skip": None}


def _missing_summary(root: Path, output_path: Path) -> dict[str, Any]:
    return {
        "stage": "bridgedata_v2_user_subset_ingestion_step22",
        "pending_user_data": True,
        "user_subset_exists": False,
        "manifest_exists": False,
        "generated_manifest_exists": False,
        "manifest_source": "missing_user_subset",
        "manifest_path": str(output_path),
        "num_manifest_records": 0,
        "num_valid_trajectories": 0,
        "num_skipped_trajectories": 0,
        "skipped_trajectories": [],
        "warnings": [f"user subset directory is missing: {root}"],
        "download_performed": False,
    }


def _find_image_dir(traj_dir: Path) -> Path | None:
    for name in ["images", "frames"]:
        candidate = traj_dir / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _first_existing_relative(root: Path, traj_dir: Path, names: list[str]) -> str | None:
    for name in names:
        candidate = traj_dir / name
        if candidate.exists() and candidate.is_file():
            return str(candidate.relative_to(root)).replace("\\", "/")
    return None
