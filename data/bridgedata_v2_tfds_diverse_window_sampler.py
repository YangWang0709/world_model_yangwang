"""Trajectory-diverse window sampling for Step30A BridgeData TFDS sanity."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from data.bridgedata_v2_tfds_window_sampler import read_window_manifest_jsonl, validate_window_record


def select_trajectory_diverse_windows(
    windows: list[dict[str, Any]],
    target_count: int = 64,
    hard_cap_windows: int = 128,
    max_windows_per_trajectory_soft_cap: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_count < 1:
        raise ValueError("target_count must be positive")
    if hard_cap_windows < 1:
        raise ValueError("hard_cap_windows must be positive")
    if max_windows_per_trajectory_soft_cap < 1:
        raise ValueError("max_windows_per_trajectory_soft_cap must be positive")

    target = min(int(target_count), int(hard_cap_windows), len(windows))
    groups = _group_windows(windows)
    selected: list[dict[str, Any]] = []
    used_sample_ids: set[str] = set()
    per_trajectory_counts: Counter[str] = Counter()

    _round_robin_add(
        groups,
        selected,
        used_sample_ids,
        per_trajectory_counts,
        target,
        max_windows_per_trajectory_soft_cap,
    )
    if len(selected) < target:
        _round_robin_add(groups, selected, used_sample_ids, per_trajectory_counts, target, None)

    for rank, record in enumerate(selected):
        record["step30a_selected_rank"] = rank

    selected_trajectories = sorted({str(record["trajectory_id"]) for record in selected})
    summary = {
        "num_available_windows": len(windows),
        "num_available_trajectories": len(groups),
        "num_selected_windows": len(selected),
        "num_selected_trajectories": len(selected_trajectories),
        "target_count": int(target_count),
        "effective_target_count": int(target),
        "hard_cap_windows": int(hard_cap_windows),
        "max_windows_per_trajectory_soft_cap": int(max_windows_per_trajectory_soft_cap),
        "trajectory_window_counts": dict(sorted(per_trajectory_counts.items())),
        "selected_trajectories": selected_trajectories,
        "selection_strategy": "round_robin_trajectory_diverse",
        "fallback_used": len(selected) < int(target_count),
        "fallback_reason": None if len(selected) >= int(target_count) else "not enough available windows",
        "new_tfds_shard_downloaded": False,
        "download_performed": False,
    }
    return selected, summary


def sample_trajectory_diverse_windows_from_manifest(
    input_manifest_jsonl: str | Path,
    output_selected_jsonl: str | Path,
    output_selection_summary_json: str | Path,
    target_count: int = 64,
    hard_cap_windows: int = 128,
    max_windows_per_trajectory_soft_cap: int = 16,
) -> dict[str, Any]:
    windows = read_window_manifest_jsonl(input_manifest_jsonl)
    selected, summary = select_trajectory_diverse_windows(
        windows,
        target_count=target_count,
        hard_cap_windows=hard_cap_windows,
        max_windows_per_trajectory_soft_cap=max_windows_per_trajectory_soft_cap,
    )
    _write_jsonl(selected, output_selected_jsonl)
    _write_json(summary, output_selection_summary_json)
    return summary


def _group_windows(windows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(windows):
        validate_window_record(record, line_number=index + 1)
        groups.setdefault(str(record["trajectory_id"]), []).append(dict(record))
    for records in groups.values():
        records.sort(key=lambda item: str(item["sample_id"]))
    return dict(sorted(groups.items()))


def _round_robin_add(
    groups: dict[str, list[dict[str, Any]]],
    selected: list[dict[str, Any]],
    used_sample_ids: set[str],
    per_trajectory_counts: Counter[str],
    target: int,
    soft_cap: int | None,
) -> None:
    cursors = {trajectory_id: 0 for trajectory_id in groups}
    while len(selected) < target:
        progressed = False
        for trajectory_id, records in groups.items():
            if soft_cap is not None and per_trajectory_counts[trajectory_id] >= soft_cap:
                continue
            cursor = cursors[trajectory_id]
            while cursor < len(records) and str(records[cursor]["sample_id"]) in used_sample_ids:
                cursor += 1
            cursors[trajectory_id] = cursor
            if cursor >= len(records):
                continue
            candidate = dict(records[cursor])
            cursors[trajectory_id] += 1
            sample_id = str(candidate["sample_id"])
            selected.append(candidate)
            used_sample_ids.add(sample_id)
            per_trajectory_counts[trajectory_id] += 1
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break


def _write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_window_record(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return output


def _write_json(payload: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output
