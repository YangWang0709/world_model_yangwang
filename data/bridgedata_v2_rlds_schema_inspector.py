"""Inspect BridgeData V2 RLDS-like episode schemas without polluting env_isaaclab."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from data.bridgedata_v2_rlds_field_resolver import resolve_rlds_fields


def inspect_rlds_like_episodes(
    episodes: list[dict[str, Any]],
    *,
    min_trajectory_len: int = 24,
    max_episodes_to_scan: int = 50,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    image_fields: set[str] = set()
    action_fields: set[str] = set()
    language_fields: set[str] = set()
    goal_fields: set[str] = set()
    proprio_fields: set[str] = set()
    for index, episode in enumerate(episodes[:max_episodes_to_scan]):
        flattened = flatten_episode_keys(episode)
        length = infer_episode_length(episode)
        for key in flattened:
            lower = key.lower()
            if "image" in lower and "goal" not in lower:
                image_fields.add(key)
            if "action" in lower:
                action_fields.add(key)
            if "language" in lower or "instruction" in lower:
                language_fields.add(key)
            if "goal" in lower:
                goal_fields.add(key)
            if "proprio" in lower or "state" in lower:
                proprio_fields.add(key)
        summaries.append(
            {
                "episode_index": index,
                "trajectory_id": f"tfds_episode_{index:06d}",
                "num_steps": length,
                "can_build_16_4_4_windows": length >= min_trajectory_len,
                "field_paths": sorted(flattened),
            }
        )
    lengths = [item["num_steps"] for item in summaries]
    candidate_fields = {
        "image_fields": sorted(image_fields),
        "action_fields": sorted(action_fields),
        "language_fields": sorted(language_fields),
        "goal_fields": sorted(goal_fields),
        "proprio_fields": sorted(proprio_fields),
    }
    return {
        "stage": "bridgedata_v2_rlds_schema_inspection",
        "tfds_env_ok": True,
        "dataset_root_exists": True,
        "num_episodes_scanned": len(summaries),
        "episode_length_min": min(lengths) if lengths else None,
        "episode_length_mean": mean(lengths) if lengths else None,
        "episode_length_max": max(lengths) if lengths else None,
        "candidate_fields": candidate_fields,
        "field_groups": classify_rlds_field_groups(set().union(*(set(item["field_paths"]) for item in summaries)) if summaries else set()),
        "resolved_fields": resolve_rlds_fields(candidate_fields),
        "episode_summaries": summaries,
        "can_build_16_4_4_windows": any(item["can_build_16_4_4_windows"] for item in summaries),
        "safe_stop": False,
        "reason": None,
        "save_video_tensors": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
    }


def inspect_tfds_dataset_root(
    dataset_root: str | Path,
    *,
    max_episodes_to_scan: int = 50,
    min_trajectory_len: int = 24,
) -> dict[str, Any]:
    root = Path(dataset_root)
    if not root.exists():
        return safe_stop_schema_summary(False, False, f"TFDS dataset root does not exist: {root}")
    try:
        import tensorflow_datasets as tfds
    except Exception as exc:
        return safe_stop_schema_summary(False, True, f"tensorflow_datasets unavailable in TFDS env: {exc}")
    try:
        import tensorflow as tf

        builder = tfds.builder_from_directory(str(root))
        read_instruction = tfds.core.ReadInstruction("train", from_=0, to=max_episodes_to_scan, unit="abs")
        dataset = builder.as_dataset(split=read_instruction, shuffle_files=False)
        episode_summaries: list[dict[str, Any]] = []
        all_field_paths: set[str] = set()
        for episode_index, episode in enumerate(dataset.take(max_episodes_to_scan)):
            steps = episode.get("steps") if isinstance(episode, dict) else None
            field_paths = _field_paths_from_tf_episode(episode)
            all_field_paths.update(field_paths)
            num_steps = _count_tf_steps(steps, tf) if steps is not None else 0
            episode_summaries.append(
                {
                    "episode_index": episode_index,
                    "trajectory_id": f"tfds_episode_{episode_index:06d}",
                    "num_steps": num_steps,
                    "can_build_16_4_4_windows": num_steps >= min_trajectory_len,
                    "field_paths": sorted(field_paths),
                }
            )
        lengths = [item["num_steps"] for item in episode_summaries]
        candidate_fields = _candidate_fields_from_paths(all_field_paths)
        summary = {
            "stage": "bridgedata_v2_rlds_schema_inspection",
            "tfds_env_ok": True,
            "dataset_root_exists": True,
            "num_episodes_scanned": len(episode_summaries),
            "episode_length_min": min(lengths) if lengths else None,
            "episode_length_mean": mean(lengths) if lengths else None,
            "episode_length_max": max(lengths) if lengths else None,
            "candidate_fields": candidate_fields,
            "field_groups": classify_rlds_field_groups(all_field_paths),
            "resolved_fields": resolve_rlds_fields(candidate_fields),
            "episode_summaries": episode_summaries,
            "can_build_16_4_4_windows": any(item["can_build_16_4_4_windows"] for item in episode_summaries),
            "safe_stop": False,
            "reason": None,
            "save_video_tensors": False,
            "training_performed": False,
            "token_extraction_performed": False,
            "importance_generation_performed": False,
        }
        summary.update({"dataset_root": str(root), "tfds_env_ok": True, "dataset_root_exists": True})
        if not summary["can_build_16_4_4_windows"]:
            summary.update({"safe_stop": True, "reason": "No scanned episode has enough steps for a 16/4/4 window."})
        return summary
    except Exception as exc:
        return safe_stop_schema_summary(True, True, f"Could not inspect TFDS/RLDS dataset root: {exc}", dataset_root=str(root))


def safe_stop_schema_summary(tfds_env_ok: bool, dataset_root_exists: bool, reason: str, *, dataset_root: str | None = None) -> dict[str, Any]:
    return {
        "stage": "bridgedata_v2_rlds_schema_inspection",
        "tfds_env_ok": tfds_env_ok,
        "dataset_root_exists": dataset_root_exists,
        "dataset_root": dataset_root,
        "num_episodes_scanned": 0,
        "episode_length_min": None,
        "episode_length_mean": None,
        "episode_length_max": None,
        "candidate_fields": {
            "image_fields": [],
            "action_fields": [],
            "language_fields": [],
            "goal_fields": [],
            "proprio_fields": [],
        },
        "field_groups": {
            "image_tensor_fields": [],
            "image_flag_fields": [],
            "action_tensor_fields": [],
            "language_text_fields": [],
            "language_embedding_fields": [],
            "goal_fields": [],
        },
        "resolved_fields": resolve_rlds_fields(
            {"image_fields": [], "action_fields": [], "language_fields": [], "goal_fields": [], "proprio_fields": []}
        ),
        "episode_summaries": [],
        "can_build_16_4_4_windows": False,
        "safe_stop": True,
        "reason": reason,
        "save_video_tensors": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
    }


def _field_paths_from_tf_episode(episode: dict[str, Any]) -> set[str]:
    fields: set[str] = set()
    if not isinstance(episode, dict):
        return fields
    for key, value in episode.items():
        if key == "steps" and hasattr(value, "element_spec"):
            fields.update(_flatten_tf_spec(value.element_spec, "steps"))
        elif isinstance(value, dict):
            fields.update(_flatten_tf_spec(value, str(key)))
        else:
            fields.add(str(key))
    return fields


def _flatten_tf_spec(value: Any, prefix: str) -> set[str]:
    if isinstance(value, dict):
        fields: set[str] = set()
        for key, child in value.items():
            fields.update(_flatten_tf_spec(child, f"{prefix}/{key}"))
        return fields
    return {prefix}


def _count_tf_steps(steps: Any, tf: Any) -> int:
    try:
        cardinality = int(steps.cardinality().numpy())
        if cardinality >= 0:
            return cardinality
    except Exception:
        pass
    try:
        count = steps.reduce(tf.constant(0, dtype=tf.int64), lambda total, _: total + 1)
        return int(count.numpy())
    except Exception:
        return 0


def _candidate_fields_from_paths(paths: set[str]) -> dict[str, list[str]]:
    image_fields: set[str] = set()
    action_fields: set[str] = set()
    language_fields: set[str] = set()
    goal_fields: set[str] = set()
    proprio_fields: set[str] = set()
    for key in paths:
        lower = key.lower()
        if "image" in lower and "goal" not in lower:
            image_fields.add(key)
        if "action" in lower:
            action_fields.add(key)
        if "language" in lower or "instruction" in lower:
            language_fields.add(key)
        if "goal" in lower:
            goal_fields.add(key)
        if "proprio" in lower or "state" in lower:
            proprio_fields.add(key)
    return {
        "image_fields": sorted(image_fields),
        "action_fields": sorted(action_fields),
        "language_fields": sorted(language_fields),
        "goal_fields": sorted(goal_fields),
        "proprio_fields": sorted(proprio_fields),
    }


def classify_rlds_field_groups(paths: set[str]) -> dict[str, list[str]]:
    image_tensor_fields: set[str] = set()
    image_flag_fields: set[str] = set()
    action_tensor_fields: set[str] = set()
    language_text_fields: set[str] = set()
    language_embedding_fields: set[str] = set()
    goal_fields: set[str] = set()
    for key in paths:
        lower = key.lower()
        if "episode_metadata/has_image_" in lower:
            image_flag_fields.add(key)
        elif key.startswith("steps/observation/") and "image" in lower and "goal" not in lower:
            image_tensor_fields.add(key)
        if key == "steps/action" or lower.endswith("/action"):
            action_tensor_fields.add(key)
        if "language" in lower or "instruction" in lower:
            if "embedding" in lower:
                language_embedding_fields.add(key)
            else:
                language_text_fields.add(key)
        if "goal" in lower:
            goal_fields.add(key)
    return {
        "image_tensor_fields": sorted(image_tensor_fields),
        "image_flag_fields": sorted(image_flag_fields),
        "action_tensor_fields": sorted(action_tensor_fields),
        "language_text_fields": sorted(language_text_fields),
        "language_embedding_fields": sorted(language_embedding_fields),
        "goal_fields": sorted(goal_fields),
    }


def flatten_episode_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        if value.get("__array__") is True:
            return {prefix} if prefix else set()
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            keys.update(flatten_episode_keys(child, child_prefix))
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        for key in flatten_episode_keys(value[0], prefix):
            keys.add(key)
    else:
        keys.add(prefix)
    return {key for key in keys if key}


def infer_episode_length(episode: dict[str, Any]) -> int:
    steps = episode.get("steps")
    if isinstance(steps, list):
        return len(steps)
    if isinstance(steps, dict):
        candidates = [_first_dim(value) for value in steps.values()]
        candidates = [item for item in candidates if item is not None]
        return max(candidates) if candidates else 0
    return _first_dim(episode) or 0


def _first_dim(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, dict):
        if value.get("__array__") is True and isinstance(value.get("shape"), list) and value["shape"]:
            return int(value["shape"][0])
        values = [_first_dim(child) for child in value.values()]
        values = [item for item in values if item is not None]
        return max(values) if values else None
    return None


def to_jsonable_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable_metadata(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable_metadata(item) for item in value]
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {"__array__": True, "shape": [int(dim) for dim in shape], "dtype": str(dtype)}
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")[:200]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    return str(type(value).__name__)


def write_schema_summary(summary: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
