"""Manifest helpers for Step24 BridgeData TFDS token smoke artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TOKEN_SHAPE_KEYS = (
    "context_token_shape",
    "current_token_shape",
    "future_token_shape",
)


def write_token_manifest_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_token_manifest_record(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return output


def read_token_manifest_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid token manifest JSON on line {line_number}: {exc}") from exc
            validate_token_manifest_record(record)
            records.append(record)
    return records


def validate_token_manifest_record(record: dict[str, Any]) -> bool:
    for key in REQUIRED_TOKEN_SHAPE_KEYS:
        shape = record.get(key)
        if not _valid_shape(shape):
            raise ValueError(f"{key} must be a non-empty integer shape, got {shape!r}")
    for flag in ("action_used_as_input", "language_used_as_input", "goal_used_as_input"):
        if bool(record.get(flag)):
            raise ValueError(f"{flag} must be false for Step24")
    image_field = record.get("image_field")
    if image_field != "steps/observation/image_0":
        raise ValueError(f"unexpected image_field {image_field!r}")
    return True


def summarize_token_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_windows_tokenized": 0,
            "num_token_artifacts": 0,
            "context_token_shape_example": None,
            "current_token_shape_example": None,
            "future_token_shape_example": None,
        }
    for record in records:
        validate_token_manifest_record(record)
    first = records[0]
    return {
        "num_windows_tokenized": len(records),
        "num_token_artifacts": len({record.get("token_artifact_path") for record in records}),
        "context_token_shape_example": first["context_token_shape"],
        "current_token_shape_example": first["current_token_shape"],
        "future_token_shape_example": first["future_token_shape"],
    }


def _valid_shape(shape: Any) -> bool:
    return isinstance(shape, list) and bool(shape) and all(isinstance(dim, int) and dim > 0 for dim in shape)
