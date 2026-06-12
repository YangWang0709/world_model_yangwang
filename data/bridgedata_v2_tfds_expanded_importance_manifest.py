"""Expanded importance manifest helpers for Step29 train/val runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.bridgedata_v2_tfds_importance_manifest import validate_importance_manifest_record


EXPECTED_CONTEXT_IMPORTANCE_SHAPE = [16, 392]


def read_expanded_importance_manifest(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            validate_expanded_importance_record(record)
            records.append(record)
    return records


def write_expanded_importance_manifest(records: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_expanded_importance_record(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return output


def enrich_importance_manifest_with_split(manifest_path: str | Path, split_json_path: str | Path) -> list[dict[str, Any]]:
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
            record["current_importance_generated"] = False
            validate_expanded_importance_record(record)
            records.append(record)
    write_expanded_importance_manifest(records, manifest_path)
    return records


def summarize_expanded_importance_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    for record in records:
        validate_expanded_importance_record(record)
        split_counts[str(record.get("trainval_split"))] = split_counts.get(str(record.get("trainval_split")), 0) + 1
    return {
        "num_importance_records": len(records),
        "split_counts": split_counts,
        "context_importance_shape_example": records[0]["context_importance_shape"] if records else None,
        "current_importance_generated": False,
    }


def validate_expanded_importance_record(record: dict[str, Any]) -> bool:
    validate_importance_manifest_record(record)
    if list(record.get("context_importance_shape") or []) != EXPECTED_CONTEXT_IMPORTANCE_SHAPE:
        raise ValueError("context_importance_shape mismatch")
    if record.get("trainval_split") not in ("train", "val"):
        raise ValueError("trainval_split must be train or val")
    if bool(record.get("current_importance_generated", False)):
        raise ValueError("current importance must not be generated")
    return True

