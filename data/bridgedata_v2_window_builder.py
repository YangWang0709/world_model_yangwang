"""Build LongContextSample-compatible BridgeData V2 window records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.bridgedata_v2_manifest_schema import normalize_bridgedata_manifest_record
from data.long_context_dataset_schema import validate_long_context_sample
from data.long_context_window_spec import LongContextWindowSpec


def build_bridgedata_windows_from_manifest(
    records: list[dict[str, Any]],
    spec: LongContextWindowSpec,
    *,
    max_trajectories: int | None = None,
    max_windows: int | None = None,
) -> list[dict[str, Any]]:
    spec.validate()
    windows: list[dict[str, Any]] = []
    selected_records = records[:max_trajectories] if max_trajectories is not None else records
    for record in selected_records:
        normalized = normalize_bridgedata_manifest_record(record)
        trajectory_windows = spec.generate_window_indices(normalized["num_frames"])
        for window_index, window in enumerate(trajectory_windows):
            sample = _make_window_record(normalized, window, window_index)
            validate_long_context_sample(sample, strict=False)
            windows.append(sample)
            if max_windows is not None and len(windows) >= max_windows:
                return windows
    return windows


def write_window_manifest_jsonl(path: str | Path, windows: list[dict[str, Any]]) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for window in windows:
            validate_long_context_sample(window, strict=False)
            handle.write(json.dumps(window, sort_keys=True) + "\n")
    return manifest_path


def load_window_manifest_jsonl(path: str | Path) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                window = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid window JSON on line {line_number}: {exc}") from exc
            validate_long_context_sample(window, strict=False)
            windows.append(window)
    return windows


def summarize_window_records(windows: list[dict[str, Any]]) -> dict[str, Any]:
    if not windows:
        return {
            "num_windows": 0,
            "num_trajectories": 0,
            "context_len": None,
            "current_len": None,
            "future_len": None,
            "has_frame_path_refs": False,
            "action_used_as_input": False,
            "language_used_as_input": False,
            "goal_image_used_as_input": False,
        }
    for window in windows:
        validate_long_context_sample(window, strict=False)
    trajectory_ids = sorted({window["trajectory_id"] for window in windows})
    first = windows[0]
    return {
        "num_windows": len(windows),
        "num_trajectories": len(trajectory_ids),
        "context_len": len(first["context_frame_indices"]),
        "current_len": len(first["current_frame_indices"]),
        "future_len": len(first["future_frame_indices"]),
        "has_frame_path_refs": any(
            bool(window.get("context_frame_paths") or window.get("current_frame_paths") or window.get("future_frame_paths"))
            for window in windows
        ),
        "action_used_as_input": any(window["metadata"].get("use_action_as_input") for window in windows),
        "language_used_as_input": any(window["metadata"].get("use_language_as_input") for window in windows),
        "goal_image_used_as_input": any(window["metadata"].get("use_goal_image_as_input") for window in windows),
    }


def summarize_manifest_window_build(
    records: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    spec: LongContextWindowSpec,
) -> dict[str, Any]:
    normalized_records = [normalize_bridgedata_manifest_record(record) for record in records]
    valid_records = [record for record in normalized_records if record["num_frames"] >= spec.required_total_frames]
    skipped = [
        {
            "trajectory_id": record["trajectory_id"],
            "num_frames": record["num_frames"],
            "reason": f"num_frames < required_total_frames ({spec.required_total_frames})",
        }
        for record in normalized_records
        if record["num_frames"] < spec.required_total_frames
    ]
    window_summary = summarize_window_records(windows)
    return {
        "num_trajectories": len(normalized_records),
        "num_valid_trajectories": len(valid_records),
        "num_skipped_trajectories": len(skipped),
        "skipped_trajectories": skipped,
        "num_windows": len(windows),
        "context_len": spec.context_len,
        "current_len": spec.current_len,
        "future_len": spec.future_len,
        "required_total_frames": spec.required_total_frames,
        "window_summary": window_summary,
    }


def _make_window_record(
    record: dict[str, Any],
    window: dict[str, list[int]],
    window_index: int,
) -> dict[str, Any]:
    trajectory_id = record["trajectory_id"]
    sample = {
        "dataset_name": "BridgeData V2",
        "split": record["split"],
        "trajectory_id": trajectory_id,
        "sample_id": f"{trajectory_id}_window_{window_index:06d}",
        "context_video": None,
        "current_video": None,
        "future_video": None,
        "context_frame_indices": list(window["context_frame_indices"]),
        "current_frame_indices": list(window["current_frame_indices"]),
        "future_frame_indices": list(window["future_frame_indices"]),
        "context_frame_paths": _paths_for_indices(record, window["context_frame_indices"]),
        "current_frame_paths": _paths_for_indices(record, window["current_frame_indices"]),
        "future_frame_paths": _paths_for_indices(record, window["future_frame_indices"]),
        "actions": None,
        "endeffector": None,
        "language_instruction": record.get("language_instruction"),
        "goal_image": None,
        "goal_image_path": record.get("goal_image_path"),
        "camera_names": list(record.get("camera_names") or []),
        "metadata": {
            "source_dataset": "BridgeData V2",
            "source_split": record["split"],
            "source_trajectory_id": trajectory_id,
            "window_builder_step": "step20",
            "action_metadata_available": bool(record.get("actions_path") or record.get("actions") is not None),
            "actions_path": record.get("actions_path"),
            "inline_actions_available": record.get("actions") is not None,
            "language_metadata_available": record.get("language_instruction") is not None,
            "goal_image_metadata_available": record.get("goal_image_path") is not None,
            "goal_image_path": record.get("goal_image_path"),
            "task_id": record.get("task_id"),
            "environment_id": record.get("environment_id"),
            "use_action_as_input": False,
            "use_language_as_input": False,
            "use_goal_image_as_input": False,
            "no_video_tensor_saved": True,
            "record_metadata": record.get("metadata") or {},
        },
    }
    return sample


def _paths_for_indices(record: dict[str, Any], indices: list[int]) -> list[str]:
    frame_paths = record.get("frame_paths")
    if frame_paths:
        return [frame_paths[index] for index in indices if index < len(frame_paths)]
    image_dir = record.get("image_dir")
    if image_dir:
        return [f"{image_dir}/frame_{index:06d}.jpg" for index in indices]
    return []
