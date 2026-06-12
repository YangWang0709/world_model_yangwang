"""Manifest helpers for Step32 BridgeData longer-horizon windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STAGE = "bridgedata_v2_tfds_longer_horizon_step32"
CONTEXT_LEN = 16
CURRENT_LEN = 4
FUTURE_LEN = 4


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL record on line {line_number}: {exc}") from exc
            records.append(record)
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_longer_horizon_window(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return output


def validate_longer_horizon_window(record: dict[str, Any]) -> bool:
    required = (
        "sample_id",
        "trajectory_id",
        "horizon_gap",
        "total_span",
        "context_frame_indices",
        "current_frame_indices",
        "gap_frame_indices",
        "future_frame_indices",
    )
    for key in required:
        if key not in record:
            raise ValueError(f"Step32 window missing {key}")
    gap = int(record["horizon_gap"])
    context = _indices(record["context_frame_indices"], CONTEXT_LEN, "context_frame_indices")
    current = _indices(record["current_frame_indices"], CURRENT_LEN, "current_frame_indices")
    future = _indices(record["future_frame_indices"], FUTURE_LEN, "future_frame_indices")
    gap_indices = _indices(record["gap_frame_indices"], gap, "gap_frame_indices")
    if int(record["total_span"]) != CONTEXT_LEN + CURRENT_LEN + gap + FUTURE_LEN:
        raise ValueError("total_span does not match Step32 lengths")
    expected_context = list(range(context[0], context[0] + CONTEXT_LEN))
    expected_current = list(range(context[0] + CONTEXT_LEN, context[0] + CONTEXT_LEN + CURRENT_LEN))
    expected_gap = list(range(context[0] + CONTEXT_LEN + CURRENT_LEN, context[0] + CONTEXT_LEN + CURRENT_LEN + gap))
    expected_future = list(
        range(
            context[0] + CONTEXT_LEN + CURRENT_LEN + gap,
            context[0] + CONTEXT_LEN + CURRENT_LEN + gap + FUTURE_LEN,
        )
    )
    if context != expected_context or current != expected_current:
        raise ValueError("context/current indices are not contiguous Step32 inputs")
    if gap_indices != expected_gap:
        raise ValueError("gap indices must be contiguous and excluded from input/target")
    if future != expected_future:
        raise ValueError("future indices are not the Step32 target indices")
    input_or_target = set(context) | set(current) | set(future)
    if set(gap_indices) & input_or_target:
        raise ValueError("gap frames must not overlap input or target frames")
    metadata = record.get("metadata") or {}
    for flag in ("use_action_as_input", "use_language_as_input", "use_goal_image_as_input"):
        if bool(metadata.get(flag)):
            raise ValueError(f"{flag} must remain false for Step32")
    return True


def summarize_horizon_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        validate_longer_horizon_window(record)
    by_gap: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_gap.setdefault(int(record["horizon_gap"]), []).append(record)
    return {
        "stage": STAGE,
        "num_windows": len(records),
        "horizons": {
            f"gap{gap}": {
                "selected_windows": len(items),
                "num_trajectories": len({str(item["trajectory_id"]) for item in items}),
                "sample_ids_unique": len({str(item["sample_id"]) for item in items}) == len(items),
            }
            for gap, items in sorted(by_gap.items())
        },
        "action_language_goal_as_metadata_only": True,
    }


def _indices(value: Any, expected_len: int, key: str) -> list[int]:
    indices = [int(item) for item in value]
    if len(indices) != expected_len:
        raise ValueError(f"{key} length {len(indices)} != expected {expected_len}")
    return indices
