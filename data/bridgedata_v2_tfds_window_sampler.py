"""Window sampling helpers for Step29 BridgeData TFDS train/val sanity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_window_manifest_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            validate_window_record(record, line_number=line_number)
            records.append(record)
    return records


def write_window_manifest_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_window_record(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return output


def select_diverse_windows(
    windows: list[dict[str, Any]],
    target_num_windows: int,
    max_windows_hard_cap: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_num_windows < 1:
        raise ValueError("target_num_windows must be positive")
    target = min(int(target_num_windows), int(max_windows_hard_cap))
    if len(windows) < target:
        target = len(windows)
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in windows:
        groups.setdefault(str(record["trajectory_id"]), []).append(record)
    for records in groups.values():
        records.sort(key=lambda item: str(item["sample_id"]))

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    group_order = sorted(groups)
    cursor = {trajectory_id: 0 for trajectory_id in group_order}
    while len(selected) < target:
        progressed = False
        for trajectory_id in group_order:
            records = groups[trajectory_id]
            index = cursor[trajectory_id]
            if index >= len(records):
                continue
            candidate = dict(records[index])
            cursor[trajectory_id] += 1
            sample_id = str(candidate["sample_id"])
            if sample_id in used_ids:
                continue
            candidate["step29_selected_rank"] = len(selected)
            selected.append(candidate)
            used_ids.add(sample_id)
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break

    selected_trajectories = sorted({str(record["trajectory_id"]) for record in selected})
    summary = {
        "num_available_windows": len(windows),
        "num_available_trajectories": len(groups),
        "num_selected_windows": len(selected),
        "num_selected_trajectories": len(selected_trajectories),
        "target_num_windows": int(target_num_windows),
        "used_optional_64": int(target_num_windows) > 32 and len(selected) > 32,
        "fallback_reason": None if len(selected) == int(target_num_windows) else "not enough available windows",
        "selected_trajectories": selected_trajectories,
        "download_performed": False,
    }
    return selected, summary


def sample_windows_from_manifest(
    input_manifest_jsonl: str | Path,
    output_selected_jsonl: str | Path,
    target_num_windows: int = 32,
    max_windows_hard_cap: int = 64,
) -> dict[str, Any]:
    windows = read_window_manifest_jsonl(input_manifest_jsonl)
    selected, summary = select_diverse_windows(
        windows,
        target_num_windows=target_num_windows,
        max_windows_hard_cap=max_windows_hard_cap,
    )
    write_window_manifest_jsonl(selected, output_selected_jsonl)
    return summary


def validate_window_record(record: dict[str, Any], line_number: int | None = None) -> bool:
    prefix = f"line {line_number}: " if line_number is not None else ""
    for key in ("sample_id", "trajectory_id", "context_frame_indices", "current_frame_indices", "future_frame_indices"):
        if key not in record:
            raise ValueError(f"{prefix}window record missing {key}")
    expected_lengths = {
        "context_frame_indices": 16,
        "current_frame_indices": 4,
        "future_frame_indices": 4,
    }
    for key, expected in expected_lengths.items():
        if len(list(record.get(key) or [])) != expected:
            raise ValueError(f"{prefix}{key} must have length {expected}")
    return True

