"""Inspect user-provided BridgeData V2 tiny-subset directories for Step22."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ACTION_FILENAMES = {"actions.npy", "actions.json"}
GOAL_FILENAMES = {"goal.jpg", "goal.jpeg", "goal.png"}


def inspect_user_subset_directory(
    user_subset_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    max_trajectories: int = 10,
) -> dict[str, Any]:
    """Inspect layout without reading image/action payloads."""

    root = Path(user_subset_root)
    manifest = Path(manifest_path) if manifest_path is not None else root / "manifest.jsonl"
    if not root.exists():
        return {
            "stage": "bridgedata_v2_user_subset_ingestion_step22",
            "user_subset_root": str(root),
            "user_subset_exists": False,
            "manifest_exists": False,
            "pending_user_data": True,
            "real_format_validated": False,
            "reason": "User subset directory does not exist yet.",
            "trajectory_count_estimate": 0,
            "trajectory_summaries": [],
            "has_images_likely": None,
            "has_actions_likely": None,
            "has_language_likely": None,
            "has_goal_image_likely": None,
            "download_performed": False,
            "training_performed": False,
            "token_extraction_performed": False,
            "importance_generation_performed": False,
            "safety_gate_pass": True,
        }
    if not root.is_dir():
        return {
            "stage": "bridgedata_v2_user_subset_ingestion_step22",
            "user_subset_root": str(root),
            "user_subset_exists": False,
            "manifest_exists": False,
            "pending_user_data": True,
            "real_format_validated": False,
            "reason": "User subset path exists but is not a directory.",
            "trajectory_count_estimate": 0,
            "trajectory_summaries": [],
            "has_images_likely": None,
            "has_actions_likely": None,
            "has_language_likely": None,
            "has_goal_image_likely": None,
            "download_performed": False,
            "training_performed": False,
            "token_extraction_performed": False,
            "importance_generation_performed": False,
            "safety_gate_pass": True,
        }

    trajectory_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    limited_dirs = trajectory_dirs[:max_trajectories]
    trajectory_summaries = [_inspect_trajectory_dir(root, path) for path in limited_dirs]
    detected_files = [path.name for path in sorted(root.iterdir())[:100]]
    return {
        "stage": "bridgedata_v2_user_subset_ingestion_step22",
        "user_subset_root": str(root),
        "user_subset_exists": True,
        "manifest_exists": manifest.exists() and manifest.is_file(),
        "manifest_path": str(manifest),
        "pending_user_data": False,
        "real_format_validated": False,
        "reason": None,
        "trajectory_count_estimate": len(trajectory_dirs),
        "trajectory_summaries": trajectory_summaries,
        "detected_files": detected_files,
        "has_images_likely": any(item["has_images_likely"] for item in trajectory_summaries),
        "has_actions_likely": any(item["has_actions_likely"] for item in trajectory_summaries),
        "has_language_likely": any(item["has_metadata_json"] for item in trajectory_summaries),
        "has_goal_image_likely": any(item["has_goal_image_likely"] for item in trajectory_summaries),
        "download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "safety_gate_pass": True,
    }


def write_user_subset_inspection_outputs(summary: dict[str, Any], json_path: str | Path, md_path: str | Path) -> None:
    json_output = Path(json_path)
    md_output = Path(md_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_output.write_text(render_user_subset_inspection_md(summary), encoding="utf-8")


def render_user_subset_inspection_md(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 User Subset Inspection",
            "",
            f"- user_subset_exists: `{str(summary['user_subset_exists']).lower()}`",
            f"- manifest_exists: `{str(summary['manifest_exists']).lower()}`",
            f"- pending_user_data: `{str(summary['pending_user_data']).lower()}`",
            f"- real_format_validated: `{str(summary['real_format_validated']).lower()}`",
            f"- trajectory_count_estimate: `{summary.get('trajectory_count_estimate')}`",
            f"- reason: `{summary.get('reason')}`",
            "- no download: `true`",
            "- no training: `true`",
            "- no token extraction: `true`",
            "- no importance generation: `true`",
            "",
        ]
    )


def _inspect_trajectory_dir(root: Path, traj_dir: Path) -> dict[str, Any]:
    names = {item.name for item in traj_dir.iterdir()}
    image_dir = _find_image_dir(traj_dir)
    frame_count = 0
    if image_dir is not None:
        frame_count = sum(1 for child in image_dir.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES)
    metadata = _read_metadata(traj_dir / "metadata.json")
    actions_path = _first_existing_relative(root, traj_dir, sorted(ACTION_FILENAMES))
    goal_path = _first_existing_relative(root, traj_dir, sorted(GOAL_FILENAMES))
    return {
        "trajectory_id": traj_dir.name,
        "has_images_likely": image_dir is not None and frame_count > 0,
        "image_dir": str(image_dir.relative_to(root)).replace("\\", "/") if image_dir is not None else None,
        "frame_count_estimate": frame_count,
        "has_metadata_json": "metadata.json" in names,
        "has_actions_likely": actions_path is not None,
        "actions_path": actions_path or metadata.get("actions_path"),
        "has_goal_image_likely": goal_path is not None,
        "goal_image_path": goal_path or metadata.get("goal_image_path"),
        "has_language_likely": metadata.get("language_instruction") is not None,
        "detected_files": sorted(names)[:50],
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
