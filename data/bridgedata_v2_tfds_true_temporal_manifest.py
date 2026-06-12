"""Manifest helpers for Step33A gap0 true-temporal diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STAGE = "bridgedata_v2_tfds_true_temporal_step33a"
HORIZON_GAP = 0


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL record on line {line_number}: {exc}") from exc
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return output


def select_step32_gap0_windows(
    step32_window_manifest_jsonl: str | Path,
    *,
    max_windows: int = 64,
) -> list[dict[str, Any]]:
    windows = [record for record in read_jsonl(step32_window_manifest_jsonl) if int(record.get("horizon_gap", -1)) == 0]
    selected = windows[: int(max_windows)]
    for record in selected:
        validate_gap0_window(record)
    return selected


def validate_gap0_window(record: dict[str, Any]) -> bool:
    if int(record.get("horizon_gap", -1)) != HORIZON_GAP:
        raise ValueError("Step33A only accepts Step32 gap0 windows")
    required = ("sample_id", "trajectory_id", "context_frame_indices", "current_frame_indices", "future_frame_indices")
    for key in required:
        if key not in record:
            raise ValueError(f"Step33A gap0 window missing {key}")
    if len(record["context_frame_indices"]) != 16:
        raise ValueError("Step33A context length must be 16")
    if len(record["current_frame_indices"]) != 4:
        raise ValueError("Step33A current length must be 4")
    if len(record["future_frame_indices"]) != 4:
        raise ValueError("Step33A future length must be 4")
    metadata = record.get("metadata") or {}
    for flag in ("use_action_as_input", "use_language_as_input", "use_goal_image_as_input"):
        if bool(metadata.get(flag)):
            raise ValueError(f"{flag} must remain false for Step33A")
    return True


def load_step32_gap0_splits(step32_horizon_splits_json: str | Path, valid_sample_ids: set[str]) -> list[dict[str, Any]]:
    payload = json.loads(Path(step32_horizon_splits_json).read_text(encoding="utf-8"))
    splits: list[dict[str, Any]] = []
    for split in payload.get("splits", []):
        if int(split.get("horizon_gap", -1)) != HORIZON_GAP:
            continue
        train_ids = [str(item) for item in split.get("train_sample_ids", []) if str(item) in valid_sample_ids]
        val_ids = [str(item) for item in split.get("val_sample_ids", []) if str(item) in valid_sample_ids]
        if not train_ids or not val_ids:
            continue
        copied = dict(split)
        copied["train_sample_ids"] = train_ids
        copied["val_sample_ids"] = val_ids
        copied["num_train_windows"] = len(train_ids)
        copied["num_val_windows"] = len(val_ids)
        copied["horizon_gap"] = HORIZON_GAP
        splits.append(copied)
    return splits


def summarize_gap0_windows(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        validate_gap0_window(record)
    return {
        "stage": STAGE,
        "horizon_gap": HORIZON_GAP,
        "num_windows": len(records),
        "num_trajectories": len({str(record["trajectory_id"]) for record in records}),
        "sample_ids_unique": len({str(record["sample_id"]) for record in records}) == len(records),
        "selected_windows_source": "step32_gap0",
        "action_language_goal_as_metadata_only": True,
    }
