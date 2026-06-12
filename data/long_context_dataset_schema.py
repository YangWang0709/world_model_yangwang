"""Unified schema helpers for long-context robot dataset migration dry-runs."""

from __future__ import annotations

from typing import Any


LongContextSample = dict[str, Any]

REQUIRED_FIELDS = {
    "dataset_name",
    "split",
    "trajectory_id",
    "sample_id",
    "context_frame_indices",
    "current_frame_indices",
    "future_frame_indices",
    "metadata",
}

OPTIONAL_FIELDS = {
    "context_video",
    "current_video",
    "future_video",
    "actions",
    "endeffector",
    "language_instruction",
    "goal_image",
    "camera_names",
}


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _validate_indices(sample: LongContextSample, key: str) -> None:
    value = sample.get(key)
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"{key} must be a list[int]")


def _validate_video_shape(sample: LongContextSample, video_key: str, index_key: str) -> None:
    value = sample.get(video_key)
    if value is None:
        return
    shape = _shape_of(value)
    if shape is None or len(shape) != 4:
        raise ValueError(f"{video_key} must expose shape [T, 3, H, W]")
    if shape[1] != 3:
        raise ValueError(f"{video_key} channel dimension must be 3")
    expected_t = len(sample.get(index_key, []))
    if expected_t and shape[0] != expected_t:
        raise ValueError(f"{video_key} T dimension {shape[0]} does not match {index_key} length {expected_t}")


def validate_long_context_sample(sample: LongContextSample, strict: bool = False) -> bool:
    if not isinstance(sample, dict):
        raise ValueError("sample must be a dictionary")
    missing = sorted(REQUIRED_FIELDS - set(sample))
    if missing:
        raise ValueError(f"missing required fields: {missing}")

    for key in ["dataset_name", "split", "trajectory_id", "sample_id"]:
        if not isinstance(sample.get(key), str) or not sample[key]:
            raise ValueError(f"{key} must be a non-empty string")
    for key in ["context_frame_indices", "current_frame_indices", "future_frame_indices"]:
        _validate_indices(sample, key)
    if not isinstance(sample.get("metadata"), dict):
        raise ValueError("metadata must be a dict")

    camera_names = sample.get("camera_names", [])
    if camera_names is not None and (
        not isinstance(camera_names, list) or not all(isinstance(item, str) for item in camera_names)
    ):
        raise ValueError("camera_names must be list[str] or None")
    language = sample.get("language_instruction")
    if language is not None and not isinstance(language, str):
        raise ValueError("language_instruction must be str or None")

    if strict:
        _validate_video_shape(sample, "context_video", "context_frame_indices")
        _validate_video_shape(sample, "current_video", "current_frame_indices")
        _validate_video_shape(sample, "future_video", "future_frame_indices")

    return True


def summarize_long_context_sample(sample: LongContextSample) -> dict[str, Any]:
    validate_long_context_sample(sample, strict=False)
    return {
        "dataset_name": sample["dataset_name"],
        "split": sample["split"],
        "trajectory_id": sample["trajectory_id"],
        "sample_id": sample["sample_id"],
        "context_len": len(sample["context_frame_indices"]),
        "current_len": len(sample["current_frame_indices"]),
        "future_len": len(sample["future_frame_indices"]),
        "has_context_video": sample.get("context_video") is not None,
        "has_current_video": sample.get("current_video") is not None,
        "has_future_video": sample.get("future_video") is not None,
        "has_actions": sample.get("actions") is not None,
        "has_endeffector": sample.get("endeffector") is not None,
        "has_language_instruction": sample.get("language_instruction") is not None,
        "has_goal_image": sample.get("goal_image") is not None,
        "camera_count": len(sample.get("camera_names") or []),
        "metadata_keys": sorted(sample.get("metadata", {}).keys()),
    }


def make_placeholder_long_context_sample(
    dataset_name: str,
    trajectory_id: str,
    window: dict[str, list[int]],
    split: str = "dryrun",
    sample_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> LongContextSample:
    sample = {
        "dataset_name": dataset_name,
        "split": split,
        "trajectory_id": trajectory_id,
        "sample_id": sample_id or f"{trajectory_id}_start_{window['context_frame_indices'][0]}",
        "context_video": None,
        "current_video": None,
        "future_video": None,
        "context_frame_indices": list(window["context_frame_indices"]),
        "current_frame_indices": list(window["current_frame_indices"]),
        "future_frame_indices": list(window["future_frame_indices"]),
        "actions": None,
        "endeffector": None,
        "language_instruction": None,
        "goal_image": None,
        "camera_names": [],
        "metadata": dict(metadata or {}),
    }
    validate_long_context_sample(sample, strict=False)
    return sample
