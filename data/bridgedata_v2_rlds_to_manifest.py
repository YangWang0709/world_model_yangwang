"""Convert inspected BridgeData V2 RLDS episodes into Step20 manifest records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.bridgedata_v2_manifest_schema import (
    normalize_bridgedata_manifest_record,
    write_bridgedata_manifest_jsonl,
)


def schema_summary_to_manifest_records(
    schema_summary: dict[str, Any],
    *,
    min_frames: int = 24,
    max_valid_trajectories: int = 10,
) -> list[dict[str, Any]]:
    candidate_fields = schema_summary.get("candidate_fields", {})
    image_field = _first(candidate_fields.get("image_fields", []))
    action_field = _first(candidate_fields.get("action_fields", []))
    language_field = _first(candidate_fields.get("language_fields", []))
    goal_field = _first(candidate_fields.get("goal_fields", []))
    records: list[dict[str, Any]] = []
    for item in schema_summary.get("episode_summaries", []):
        num_steps = int(item.get("num_steps") or 0)
        if num_steps < min_frames:
            continue
        episode_index = int(item.get("episode_index", len(records)))
        trajectory_id = item.get("trajectory_id") or f"tfds_episode_{episode_index:06d}"
        record = normalize_bridgedata_manifest_record(
            {
                "dataset_name": "BridgeData V2",
                "split": "train",
                "trajectory_id": trajectory_id,
                "num_frames": num_steps,
                "frame_paths": [],
                "image_dir": None,
                "camera_names": [image_field] if image_field else [],
                "actions_path": None,
                "actions": None,
                "language_instruction": item.get("language_instruction"),
                "goal_image_path": None,
                "task_id": None,
                "environment_id": None,
                "metadata": {
                    "source": "tfds_rlds_mini_shard",
                    "tfds_episode_index": episode_index,
                    "image_field": image_field,
                    "action_field": action_field,
                    "language_field": language_field,
                    "goal_field": goal_field,
                    "field_paths": item.get("field_paths", []),
                    "use_action_as_input": False,
                    "use_language_as_input": False,
                    "use_goal_image_as_input": False,
                    "save_video_tensors": False,
                },
            }
        )
        records.append(record)
        if len(records) >= max_valid_trajectories:
            break
    return records


def fake_rlds_episodes_to_manifest_records(
    episodes: list[dict[str, Any]],
    *,
    min_frames: int = 24,
    max_valid_trajectories: int = 10,
) -> list[dict[str, Any]]:
    from data.bridgedata_v2_rlds_schema_inspector import inspect_rlds_like_episodes

    schema = inspect_rlds_like_episodes(episodes, min_trajectory_len=min_frames)
    return schema_summary_to_manifest_records(schema, min_frames=min_frames, max_valid_trajectories=max_valid_trajectories)


def write_manifest_and_summary(
    records: list[dict[str, Any]],
    manifest_path: str | Path,
    summary_path: str | Path,
    *,
    schema_summary: dict[str, Any],
    min_valid_trajectories: int = 1,
) -> dict[str, Any]:
    manifest = write_bridgedata_manifest_jsonl(manifest_path, records)
    summary = {
        "stage": "bridgedata_v2_tfds_mini_manifest",
        "real_tfds_validated": len(records) >= min_valid_trajectories,
        "safe_stop": len(records) < min_valid_trajectories,
        "reason": None if len(records) >= min_valid_trajectories else "No valid TFDS/RLDS trajectories reached min_frames.",
        "manifest_path": str(manifest),
        "num_manifest_records": len(records),
        "num_valid_trajectories": len(records),
        "num_skipped_trajectories": max(0, len(schema_summary.get("episode_summaries", [])) - len(records)),
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "save_video_tensors": False,
    }
    output = Path(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _first(values: list[Any]) -> str | None:
    return str(values[0]) if values else None
