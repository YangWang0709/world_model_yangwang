"""Build Step33B gap0 windows from one BridgeData V2 TFDS shard."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from data.bridgedata_v2_tfds_longer_horizon_manifest import (
    CONTEXT_LEN,
    CURRENT_LEN,
    FUTURE_LEN,
    validate_longer_horizon_window,
    write_jsonl,
)
from data.bridgedata_v2_tfds_shard_inventory import read_shard_lengths, shard_episode_range


STAGE = "bridgedata_v2_tfds_data_diversity_step33b"


def build_gap0_windows_from_tfds_shard(
    *,
    dataset_root: str | Path,
    resolved_fields_json: str | Path,
    output_manifest_jsonl: str | Path,
    output_summary_json: str | Path,
    shard_index: int,
    target_windows: int = 64,
    min_windows: int = 32,
    max_windows_per_trajectory_soft_cap: int = 16,
    context_len: int = CONTEXT_LEN,
    current_len: int = CURRENT_LEN,
    future_len: int = FUTURE_LEN,
    horizon_gap: int = 0,
) -> dict[str, Any]:
    resolved_fields = json.loads(Path(resolved_fields_json).read_text(encoding="utf-8"))
    shard_lengths = read_shard_lengths(dataset_root)
    if not shard_lengths:
        summary = _safe_stop_summary("dataset_info shardLengths missing", shard_index, min_windows)
        return _write_summary(summary, output_summary_json)
    start, end = shard_episode_range(shard_lengths, int(shard_index))
    try:
        import tensorflow_datasets as tfds
    except Exception as exc:
        summary = _safe_stop_summary(f"tensorflow_datasets unavailable in TFDS env: {exc}", shard_index, min_windows)
        return _write_summary(summary, output_summary_json)

    builder = tfds.builder_from_directory(str(dataset_root))
    read_instruction = tfds.core.ReadInstruction("train", from_=start, to=end, unit="abs")
    dataset = builder.as_dataset(split=read_instruction, shuffle_files=False)
    candidates: list[dict[str, Any]] = []
    trajectory_summaries: list[dict[str, Any]] = []
    for local_episode_index, episode in enumerate(dataset):
        absolute_episode_index = start + local_episode_index
        steps = list(episode["steps"])
        length = len(steps)
        trajectory_id = f"shard{int(shard_index)}_tfds_episode_{absolute_episode_index:06d}"
        episode_candidates = _windows_for_episode(
            trajectory_id=trajectory_id,
            absolute_episode_index=absolute_episode_index,
            shard_index=int(shard_index),
            steps=steps,
            resolved_fields=resolved_fields,
            context_len=context_len,
            current_len=current_len,
            future_len=future_len,
            horizon_gap=horizon_gap,
        )
        candidates.extend(episode_candidates)
        trajectory_summaries.append(
            _trajectory_summary(
                trajectory_id=trajectory_id,
                absolute_episode_index=absolute_episode_index,
                shard_index=int(shard_index),
                steps=steps,
                resolved_fields=resolved_fields,
                num_candidate_windows=len(episode_candidates),
            )
        )
    selected = _select_diverse_windows(
        candidates,
        target_count=int(target_windows),
        soft_cap=int(max_windows_per_trajectory_soft_cap),
    )
    safe_stop = len(selected) < int(min_windows)
    if safe_stop:
        selected = []
    for record in selected:
        validate_longer_horizon_window(record)
    write_jsonl(selected, output_manifest_jsonl)
    summary = {
        "stage": STAGE,
        "safe_stop": bool(safe_stop),
        "reason": f"selected_windows < min_windows ({min_windows})" if safe_stop else None,
        "shard": "shard1",
        "shard_index": int(shard_index),
        "episode_start": start,
        "episode_end_exclusive": end,
        "available_episodes": max(0, end - start),
        "available_windows": len(candidates),
        "selected_windows": len(selected),
        "num_trajectories": len({str(record["trajectory_id"]) for record in selected}),
        "available_trajectories": len(trajectory_summaries),
        "trajectory_window_counts": dict(Counter(str(record["trajectory_id"]) for record in selected)),
        "trajectory_summaries": trajectory_summaries,
        "length_stats": _length_stats([int(item["episode_length"]) for item in trajectory_summaries]),
        "language_hash_counts": dict(Counter(str(item["language_hash"]) for item in trajectory_summaries)),
        "action_stats": _aggregate_action_stats(trajectory_summaries),
        "target_variant": "future_delta_last_minus_current",
        "horizon_gap": int(horizon_gap),
        "context_len": int(context_len),
        "current_len": int(current_len),
        "future_len": int(future_len),
        "action_language_goal_as_metadata_only": True,
        "raw_language_text_saved": False,
        "download_performed": False,
        "raw_zip_downloaded": False,
        "full_tfds_downloaded": False,
        "model_download_performed": False,
        "safety_gate_pass": True,
    }
    return _write_summary(summary, output_summary_json)


def build_fake_gap0_windows(
    trajectory_lengths: dict[str, int],
    *,
    shard_index: int = 1,
    target_windows: int = 64,
    min_windows: int = 32,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for episode_index, (trajectory_id, length) in enumerate(sorted(trajectory_lengths.items())):
        for start in range(0, max(0, int(length) - (CONTEXT_LEN + CURRENT_LEN + FUTURE_LEN) + 1)):
            candidates.append(
                _window_record(
                    trajectory_id=trajectory_id,
                    sample_id=f"shard{shard_index}_{trajectory_id}_window_{start:06d}",
                    start=start,
                    absolute_episode_index=episode_index,
                    shard_index=shard_index,
                    metadata_extra={"fake": True},
                )
            )
    selected = _select_diverse_windows(candidates, target_count=target_windows, soft_cap=16)
    safe_stop = len(selected) < min_windows
    if safe_stop:
        selected = []
    return selected, {
        "safe_stop": safe_stop,
        "available_windows": len(candidates),
        "selected_windows": len(selected),
        "action_language_goal_as_metadata_only": True,
    }


def _windows_for_episode(
    *,
    trajectory_id: str,
    absolute_episode_index: int,
    shard_index: int,
    steps: list[Any],
    resolved_fields: dict[str, Any],
    context_len: int,
    current_len: int,
    future_len: int,
    horizon_gap: int,
) -> list[dict[str, Any]]:
    total_span = int(context_len) + int(current_len) + int(horizon_gap) + int(future_len)
    max_start = max(0, len(steps) - total_span + 1)
    records: list[dict[str, Any]] = []
    for start in range(max_start):
        records.append(
            _window_record(
                trajectory_id=trajectory_id,
                sample_id=f"{trajectory_id}_window_{start:06d}",
                start=start,
                absolute_episode_index=absolute_episode_index,
                shard_index=shard_index,
                metadata_extra={
                    "record_metadata": {
                        "tfds_episode_index": absolute_episode_index,
                        "source": "tfds_rlds_second_shard",
                        "shard_index": shard_index,
                        "image_field": resolved_fields.get("image_field"),
                        "action_field": resolved_fields.get("action_field"),
                        "language_field": resolved_fields.get("language_field"),
                        "image_field_is_metadata_flag": False,
                        "image_field_valid": True,
                        "action_field_valid": bool(resolved_fields.get("action_field")),
                        "language_field_valid": bool(resolved_fields.get("language_field")),
                    }
                },
            )
        )
    return records


def _window_record(
    *,
    trajectory_id: str,
    sample_id: str,
    start: int,
    absolute_episode_index: int,
    shard_index: int,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = list(range(start, start + CONTEXT_LEN))
    current = list(range(start + CONTEXT_LEN, start + CONTEXT_LEN + CURRENT_LEN))
    gap_indices: list[int] = []
    future_start = start + CONTEXT_LEN + CURRENT_LEN
    future = list(range(future_start, future_start + FUTURE_LEN))
    metadata = {
        "step33b_data_diversity": True,
        "horizon_gap": 0,
        "shard_index": int(shard_index),
        "source_context_start": int(start),
        "tfds_episode_index": int(absolute_episode_index),
        "use_action_as_input": False,
        "use_language_as_input": False,
        "use_goal_image_as_input": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
    }
    metadata.update(metadata_extra or {})
    record = {
        "sample_id": sample_id,
        "trajectory_id": trajectory_id,
        "split": "train",
        "dataset_name": "BridgeData V2",
        "source_split": "train",
        "horizon_gap": 0,
        "total_span": CONTEXT_LEN + CURRENT_LEN + FUTURE_LEN,
        "context_frame_indices": context,
        "current_frame_indices": current,
        "gap_frame_indices": gap_indices,
        "future_frame_indices": future,
        "context_video": None,
        "current_video": None,
        "future_video": None,
        "actions": None,
        "language_instruction": None,
        "metadata": metadata,
    }
    validate_longer_horizon_window(record)
    return record


def _select_diverse_windows(candidates: list[dict[str, Any]], *, target_count: int, soft_cap: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        grouped[str(record["trajectory_id"])].append(record)
    for records in grouped.values():
        records.sort(key=lambda item: int(item["context_frame_indices"][0]))
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    while len(selected) < min(target_count, len(candidates)):
        progressed = False
        for trajectory_id in sorted(grouped):
            if len(selected) >= min(target_count, len(candidates)):
                break
            if counts[trajectory_id] >= soft_cap:
                continue
            records = grouped[trajectory_id]
            if counts[trajectory_id] < len(records):
                selected.append(records[counts[trajectory_id]])
                counts[trajectory_id] += 1
                progressed = True
        if not progressed:
            break
    if len(selected) < min(target_count, len(candidates)):
        used = {str(record["sample_id"]) for record in selected}
        for record in sorted(candidates, key=lambda item: (str(item["trajectory_id"]), int(item["context_frame_indices"][0]))):
            if len(selected) >= min(target_count, len(candidates)):
                break
            if str(record["sample_id"]) not in used:
                selected.append(record)
                used.add(str(record["sample_id"]))
    return selected


def _trajectory_summary(
    *,
    trajectory_id: str,
    absolute_episode_index: int,
    shard_index: int,
    steps: list[Any],
    resolved_fields: dict[str, Any],
    num_candidate_windows: int,
) -> dict[str, Any]:
    language = _first_language(steps, str(resolved_fields.get("language_field") or ""))
    action_stats = _episode_action_stats(steps, str(resolved_fields.get("action_field") or ""))
    return {
        "trajectory_id": trajectory_id,
        "tfds_episode_index": int(absolute_episode_index),
        "shard_index": int(shard_index),
        "episode_length": len(steps),
        "num_candidate_windows": int(num_candidate_windows),
        "language_hash": _hash_text(language),
        "language_present": bool(language),
        "action_stats": action_stats,
    }


def _first_language(steps: list[Any], field: str) -> str:
    if not field:
        return ""
    for step in steps[: min(4, len(steps))]:
        value = _nested_value(step, field)
        if value is None:
            continue
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        if isinstance(value, np.ndarray) and value.shape == ():
            item = value.item()
            if isinstance(item, bytes):
                return item.decode("utf-8", "replace")
            return str(item)
        return str(value)
    return ""


def _episode_action_stats(steps: list[Any], field: str) -> dict[str, Any]:
    if not field:
        return {"available": False}
    values: list[np.ndarray] = []
    for step in steps:
        value = _nested_value(step, field)
        if value is None:
            continue
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size:
            values.append(array)
    if not values:
        return {"available": False}
    stacked = np.stack(values, axis=0)
    return {
        "available": True,
        "dim": int(stacked.shape[1]),
        "mean_abs": float(np.mean(np.abs(stacked))),
        "std_mean": float(np.mean(np.std(stacked, axis=0))),
        "l2_mean": float(np.mean(np.linalg.norm(stacked, axis=1))),
    }


def _aggregate_action_stats(trajectory_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    stats = [item.get("action_stats") or {} for item in trajectory_summaries if (item.get("action_stats") or {}).get("available")]
    if not stats:
        return {"available": False}
    return {
        "available": True,
        "episodes_with_action": len(stats),
        "mean_abs_mean": float(np.mean([float(item["mean_abs"]) for item in stats])),
        "std_mean_mean": float(np.mean([float(item["std_mean"]) for item in stats])),
        "l2_mean_mean": float(np.mean([float(item["l2_mean"]) for item in stats])),
    }


def _nested_value(step: Any, field: str) -> Any:
    value = step
    parts = field.split("/")
    if parts and parts[0] == "steps":
        parts = parts[1:]
    try:
        for part in parts:
            value = value[part]
        return value
    except Exception:
        return None


def _hash_text(text: str) -> str:
    if not text:
        return "missing"
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]


def _length_stats(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"count": 0}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
    }


def _safe_stop_summary(reason: str, shard_index: int, min_windows: int) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "safe_stop": True,
        "reason": reason,
        "shard": "shard1",
        "shard_index": int(shard_index),
        "available_windows": 0,
        "selected_windows": 0,
        "min_windows": int(min_windows),
        "num_trajectories": 0,
        "action_language_goal_as_metadata_only": True,
        "raw_language_text_saved": False,
        "safety_gate_pass": True,
    }


def _write_summary(summary: dict[str, Any], path: str | Path) -> dict[str, Any]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
