"""Export tiny BridgeData TFDS/RLDS clips for Step24 token extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

EXPECTED_IMAGE_FIELD = "steps/observation/image_0"
STAGE = "bridgedata_v2_tfds_clip_export"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def export_tfds_resolved_clips_from_config(config_path: str | Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    resolved_fields_path = Path(config["input"]["resolved_fields_json"])
    window_manifest_path = Path(config["input"]["resolved_window_manifest_jsonl"])
    dataset_root = Path(config["input"]["tfds_dataset_root"])
    output_cfg = config["output"]
    clip_cache_dir = Path(output_cfg["clip_cache_dir"])
    summary_path = Path(output_cfg["clip_export_summary_json"])
    max_windows = int(config["dry_run_limits"]["max_windows"])
    image_size = int(config["clip_export"]["image_size"])

    if not resolved_fields_path.exists():
        summary = _safe_stop_summary("resolved_fields_json is missing", clip_cache_dir, max_windows)
        return _write_summary(summary, summary_path)
    if not window_manifest_path.exists():
        summary = _safe_stop_summary("resolved_window_manifest_jsonl is missing", clip_cache_dir, max_windows)
        return _write_summary(summary, summary_path)
    if not dataset_root.exists():
        summary = _safe_stop_summary(f"TFDS dataset root is missing: {dataset_root}", clip_cache_dir, max_windows)
        return _write_summary(summary, summary_path)

    resolved_fields = json.loads(resolved_fields_path.read_text(encoding="utf-8"))
    _validate_image_field(resolved_fields)
    windows = load_jsonl(window_manifest_path)[:max_windows]
    if not windows:
        summary = _safe_stop_summary("resolved window manifest has no windows", clip_cache_dir, max_windows)
        return _write_summary(summary, summary_path)

    try:
        import tensorflow_datasets as tfds
    except Exception as exc:
        summary = _safe_stop_summary(f"tensorflow_datasets unavailable in TFDS env: {exc}", clip_cache_dir, max_windows)
        return _write_summary(summary, summary_path)

    max_episode_index = max(_episode_index(window) for window in windows)
    builder = tfds.builder_from_directory(str(dataset_root))
    read_instruction = tfds.core.ReadInstruction("train", from_=0, to=max_episode_index + 1, unit="abs")
    dataset = builder.as_dataset(split=read_instruction, shuffle_files=False)
    selected_by_episode: dict[int, list[dict[str, Any]]] = {}
    for window in windows:
        selected_by_episode.setdefault(_episode_index(window), []).append(window)

    _prepare_clip_cache_dir(clip_cache_dir)
    records: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(dataset):
        episode_windows = selected_by_episode.get(episode_index, [])
        if not episode_windows:
            continue
        step_list = list(episode["steps"])
        for window in episode_windows:
            records.append(
                export_window_to_clip_cache(
                    window=window,
                    steps=step_list,
                    resolved_fields=resolved_fields,
                    clip_cache_dir=clip_cache_dir,
                    image_size=image_size,
                )
            )
        if len(records) >= len(windows):
            break

    summary = _summary_from_records(records, clip_cache_dir, max_windows, resolved_fields)
    return _write_summary(summary, summary_path)


def export_window_to_clip_cache(
    *,
    window: dict[str, Any],
    steps: list[Any],
    resolved_fields: dict[str, Any],
    clip_cache_dir: str | Path,
    image_size: int = 224,
) -> dict[str, Any]:
    _validate_image_field(resolved_fields)
    clip_dir = Path(clip_cache_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    image_field = str(resolved_fields["image_field"])
    context = _frames_for_indices(steps, window["context_frame_indices"], image_field, image_size)
    current = _frames_for_indices(steps, window["current_frame_indices"], image_field, image_size)
    future = _frames_for_indices(steps, window["future_frame_indices"], image_field, image_size)
    sample_id = str(window.get("sample_id") or f"{window['trajectory_id']}_window")
    artifact_path = clip_dir / f"{_sanitize_id(sample_id)}.npz"
    metadata = _metadata_for_window(window, resolved_fields, artifact_path)
    np.savez_compressed(
        artifact_path,
        context_video=context,
        current_video=current,
        future_video=future,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    return {
        "sample_id": sample_id,
        "trajectory_id": window.get("trajectory_id"),
        "clip_cache_path": str(artifact_path),
        "context_shape": list(context.shape),
        "current_shape": list(current.shape),
        "future_shape": list(future.shape),
        "metadata": metadata,
    }


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _validate_image_field(resolved_fields: dict[str, Any]) -> None:
    image_field = resolved_fields.get("image_field")
    if image_field != EXPECTED_IMAGE_FIELD:
        raise ValueError(f"Step24 requires image_field={EXPECTED_IMAGE_FIELD}, got {image_field!r}")
    if "episode_metadata" in str(image_field):
        raise ValueError(f"Step24 refuses metadata image flag: {image_field!r}")
    if not bool(resolved_fields.get("image_field_valid", True)):
        raise ValueError("Step24 requires image_field_valid=true")


def _episode_index(window: dict[str, Any]) -> int:
    metadata = window.get("metadata") or {}
    record_metadata = metadata.get("record_metadata") or {}
    if record_metadata.get("tfds_episode_index") is not None:
        return int(record_metadata["tfds_episode_index"])
    match = re.search(r"tfds_episode_(\d+)", str(window.get("trajectory_id", "")))
    if not match:
        raise ValueError(f"Could not infer TFDS episode index from {window.get('trajectory_id')!r}")
    return int(match.group(1))


def _frames_for_indices(steps: list[Any], indices: list[int], image_field: str, image_size: int) -> np.ndarray:
    frames = [_extract_image_frame(steps[int(index)], image_field, image_size) for index in indices]
    return np.stack(frames, axis=0)


def _extract_image_frame(step: Any, image_field: str, image_size: int) -> np.ndarray:
    value = step
    parts = image_field.split("/")
    if parts and parts[0] == "steps":
        parts = parts[1:]
    for part in parts:
        value = value[part]
    array = _to_numpy(value)
    return _resize_uint8_tchw(array, image_size=image_size)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _resize_uint8_tchw(array: np.ndarray, *, image_size: int) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim != 3:
        raise ValueError(f"image frame must be rank 3, got {tuple(image.shape)}")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"image frame must have 3 channels, got {tuple(image.shape)}")
    if np.issubdtype(image.dtype, np.floating) and float(np.nanmax(image)) <= 1.0:
        image = image * 255.0
    if image.shape[0] != image_size or image.shape[1] != image_size:
        y_index = np.linspace(0, image.shape[0] - 1, image_size).round().astype(np.int64)
        x_index = np.linspace(0, image.shape[1] - 1, image_size).round().astype(np.int64)
        image = image[y_index][:, x_index]
    image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    return np.transpose(image, (2, 0, 1))


def _metadata_for_window(window: dict[str, Any], resolved_fields: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    record_metadata = (window.get("metadata") or {}).get("record_metadata") or {}
    return {
        "sample_id": window.get("sample_id"),
        "trajectory_id": window.get("trajectory_id"),
        "split": window.get("split"),
        "clip_cache_path": str(artifact_path),
        "image_field": resolved_fields.get("image_field"),
        "image_field_is_metadata_flag": False,
        "action_field": resolved_fields.get("action_field"),
        "language_field": resolved_fields.get("language_field"),
        "goal_field": resolved_fields.get("goal_field"),
        "tfds_episode_index": record_metadata.get("tfds_episode_index"),
        "context_frame_indices": list(window.get("context_frame_indices", [])),
        "current_frame_indices": list(window.get("current_frame_indices", [])),
        "future_frame_indices": list(window.get("future_frame_indices", [])),
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
    }


def _prepare_clip_cache_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("*.npz"):
        old.unlink()


def _safe_stop_summary(reason: str, clip_cache_dir: Path, max_windows: int) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "clip_export_performed": False,
        "safe_stop": True,
        "reason": reason,
        "num_windows_requested": max_windows,
        "num_windows_exported": 0,
        "image_field": EXPECTED_IMAGE_FIELD,
        "image_field_is_metadata_flag": False,
        "context_shape": None,
        "current_shape": None,
        "future_shape": None,
        "clip_cache_dir": str(clip_cache_dir),
        "clip_cache_files": [],
        "clip_cache_total_bytes": 0,
        "download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "safety_gate_pass": True,
    }


def _summary_from_records(
    records: list[dict[str, Any]],
    clip_cache_dir: Path,
    max_windows: int,
    resolved_fields: dict[str, Any],
) -> dict[str, Any]:
    files = [Path(record["clip_cache_path"]) for record in records]
    first = records[0] if records else {}
    return {
        "stage": STAGE,
        "clip_export_performed": bool(records),
        "safe_stop": not bool(records),
        "reason": None if records else "No windows were exported.",
        "num_windows_requested": max_windows,
        "num_windows_exported": len(records),
        "image_field": resolved_fields.get("image_field"),
        "image_field_is_metadata_flag": False,
        "context_shape": first.get("context_shape"),
        "current_shape": first.get("current_shape"),
        "future_shape": first.get("future_shape"),
        "clip_cache_dir": str(clip_cache_dir),
        "clip_cache_files": [str(path) for path in files],
        "clip_cache_total_bytes": sum(path.stat().st_size for path in files if path.exists()),
        "download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "safety_gate_pass": True,
    }


def _write_summary(summary: dict[str, Any], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
