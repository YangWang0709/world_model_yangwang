"""Expanded token manifest helpers for Step29 train/val runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.bridgedata_v2_tfds_token_manifest import validate_token_manifest_record


EXPECTED_CONTEXT_SHAPE = [16, 392, 768]
EXPECTED_CURRENT_SHAPE = [4, 392, 768]
EXPECTED_FUTURE_SHAPE = [4, 392, 768]


def read_expanded_token_manifest(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            validate_expanded_token_record(record)
            records.append(record)
    return records


def write_expanded_token_manifest(records: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_expanded_token_record(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return output


def enrich_token_manifest_with_split(manifest_path: str | Path, split_json_path: str | Path) -> list[dict[str, Any]]:
    split = json.loads(Path(split_json_path).read_text(encoding="utf-8"))
    split_by_id = {sample_id: "train" for sample_id in split.get("train_sample_ids", [])}
    split_by_id.update({sample_id: "val" for sample_id in split.get("val_sample_ids", [])})
    records: list[dict[str, Any]] = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            sample_id = str(record["sample_id"])
            record["trainval_split"] = split_by_id.get(sample_id)
            validate_expanded_token_record(record)
            records.append(record)
    write_expanded_token_manifest(records, manifest_path)
    return records


def summarize_expanded_token_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    for record in records:
        validate_expanded_token_record(record)
        split_counts[str(record.get("trainval_split"))] = split_counts.get(str(record.get("trainval_split")), 0) + 1
    return {
        "num_token_records": len(records),
        "split_counts": split_counts,
        "context_token_shape_example": records[0]["context_token_shape"] if records else None,
        "current_token_shape_example": records[0]["current_token_shape"] if records else None,
        "future_token_shape_example": records[0]["future_token_shape"] if records else None,
    }


def validate_expanded_token_record(record: dict[str, Any]) -> bool:
    validate_token_manifest_record(record)
    if list(record.get("context_token_shape") or []) != EXPECTED_CONTEXT_SHAPE:
        raise ValueError("context_token_shape mismatch")
    if list(record.get("current_token_shape") or []) != EXPECTED_CURRENT_SHAPE:
        raise ValueError("current_token_shape mismatch")
    if list(record.get("future_token_shape") or []) != EXPECTED_FUTURE_SHAPE:
        raise ValueError("future_token_shape mismatch")
    if record.get("trainval_split") not in ("train", "val"):
        raise ValueError("trainval_split must be train or val")
    return True

