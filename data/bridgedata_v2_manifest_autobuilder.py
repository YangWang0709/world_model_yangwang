"""Autobuild Step20 BridgeData manifests from tiny local directory layouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.bridgedata_v2_manifest_schema import normalize_bridgedata_manifest_record, write_bridgedata_manifest_jsonl


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def build_manifest_from_directory(
    subset_dir: str | Path,
    output_manifest: str | Path,
    *,
    min_frames: int = 24,
    max_trajectories: int | None = None,
) -> dict[str, Any]:
    root = Path(subset_dir)
    output_path = Path(output_manifest)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not root.exists() or not root.is_dir():
        summary = _empty_summary(root, output_path, "subset directory missing")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return summary

    manifest = root / "manifest.jsonl"
    if manifest.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
        return {
            "manifest_built": True,
            "manifest_source": "existing_manifest_jsonl",
            "manifest_path": str(output_path),
            "num_manifest_records": sum(1 for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()),
            "num_valid_trajectories": None,
            "num_skipped_trajectories": 0,
            "skipped_trajectories": [],
            "warnings": [],
        }

    trajectory_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if max_trajectories is not None:
        trajectory_dirs = trajectory_dirs[:max_trajectories]
    for traj_dir in trajectory_dirs:
        record_or_skip = _record_from_trajectory_dir(root, traj_dir, min_frames)
        if record_or_skip.get("record") is not None:
            records.append(record_or_skip["record"])
        else:
            skipped.append(record_or_skip["skip"])
    if not trajectory_dirs:
        warnings.append("no trajectory directories found")
    write_bridgedata_manifest_jsonl(output_path, records)
    return {
        "manifest_built": bool(records),
        "manifest_source": "directory_per_trajectory",
        "manifest_path": str(output_path),
        "num_manifest_records": len(records),
        "num_valid_trajectories": len(records),
        "num_skipped_trajectories": len(skipped),
        "skipped_trajectories": skipped,
        "warnings": warnings,
    }


def _record_from_trajectory_dir(root: Path, traj_dir: Path, min_frames: int) -> dict[str, Any]:
    image_dir = _find_image_dir(traj_dir)
    frame_paths = []
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
    goal_image_path = _first_existing_relative(root, traj_dir, ["goal.jpg", "goal.jpeg", "goal.png"])
    actions_path = _first_existing_relative(root, traj_dir, ["actions.npy", "actions.json"])
    record = normalize_bridgedata_manifest_record(
        {
            "dataset_name": "BridgeData V2",
            "split": metadata.get("split", "real_tiny"),
            "trajectory_id": metadata.get("trajectory_id", traj_dir.name),
            "num_frames": len(frame_paths),
            "frame_paths": frame_paths,
            "image_dir": str(image_dir.relative_to(root)).replace("\\", "/") if image_dir else None,
            "camera_names": metadata.get("camera_names") or ["main"],
            "actions_path": actions_path,
            "language_instruction": metadata.get("language_instruction"),
            "goal_image_path": goal_image_path,
            "task_id": metadata.get("task_id"),
            "environment_id": metadata.get("environment_id"),
            "metadata": {"source": "autobuilt_step21", **metadata.get("metadata", {})},
        }
    )
    return {"record": record, "skip": None}


def _empty_summary(root: Path, output_path: Path, warning: str) -> dict[str, Any]:
    return {
        "manifest_built": False,
        "manifest_source": "missing",
        "manifest_path": str(output_path),
        "num_manifest_records": 0,
        "num_valid_trajectories": 0,
        "num_skipped_trajectories": 0,
        "skipped_trajectories": [],
        "warnings": [warning],
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
        if candidate.exists():
            return str(candidate.relative_to(root)).replace("\\", "/")
    return None
