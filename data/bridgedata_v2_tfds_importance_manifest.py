"""Manifest helpers for Step25 BridgeData TFDS importance smoke artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

EXPECTED_CONTEXT_IMPORTANCE_SHAPE = [16, 392]
EXPECTED_TEMPORAL_IMPORTANCE_SHAPE = [16]
EXPECTED_SPATIAL_IMPORTANCE_SHAPE = [392]
METHOD = "proxy_token_mse_dryrun"


def write_importance_manifest_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_importance_manifest_record(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return output


def read_importance_manifest_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid importance manifest JSON on line {line_number}: {exc}") from exc
            validate_importance_manifest_record(record)
            records.append(record)
    return records


def validate_importance_manifest_record(record: dict[str, Any]) -> bool:
    if record.get("method") != METHOD:
        raise ValueError(f"unexpected method {record.get('method')!r}")
    expected_shapes = {
        "context_importance_shape": EXPECTED_CONTEXT_IMPORTANCE_SHAPE,
        "temporal_importance_shape": EXPECTED_TEMPORAL_IMPORTANCE_SHAPE,
        "spatial_importance_shape": EXPECTED_SPATIAL_IMPORTANCE_SHAPE,
    }
    for key, expected in expected_shapes.items():
        if list(record.get(key) or []) != expected:
            raise ValueError(f"{key} {record.get(key)!r} != expected {expected!r}")
    for flag, expected in {
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
    }.items():
        if bool(record.get(flag)) != expected:
            raise ValueError(f"{flag} must be {expected}")
    return True


def validate_importance_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    artifact = _torch_load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError(f"importance artifact must be a dict: {artifact_path}")
    if artifact.get("method") != METHOD:
        raise ValueError(f"unexpected importance method {artifact.get('method')!r}")
    expected = {
        "context_importance_raw": EXPECTED_CONTEXT_IMPORTANCE_SHAPE,
        "context_importance_norm": EXPECTED_CONTEXT_IMPORTANCE_SHAPE,
        "temporal_importance": EXPECTED_TEMPORAL_IMPORTANCE_SHAPE,
        "spatial_importance": EXPECTED_SPATIAL_IMPORTANCE_SHAPE,
    }
    for key, shape in expected.items():
        value = artifact.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{key} missing or not tensor")
        if list(value.shape) != shape:
            raise ValueError(f"{key} shape {list(value.shape)} != expected {shape}")
    metadata = artifact.get("metadata") or {}
    if bool(metadata.get("action_used_as_input")):
        raise ValueError("action must not be used as Step25 input")
    return artifact


def summarize_importance_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_samples": 0,
            "num_importance_artifacts": 0,
            "context_importance_shape_example": None,
            "temporal_importance_shape_example": None,
            "spatial_importance_shape_example": None,
        }
    for record in records:
        validate_importance_manifest_record(record)
    first = records[0]
    return {
        "num_samples": len(records),
        "num_importance_artifacts": len({record["importance_artifact_path"] for record in records}),
        "context_importance_shape_example": first["context_importance_shape"],
        "temporal_importance_shape_example": first["temporal_importance_shape"],
        "spatial_importance_shape_example": first["spatial_importance_shape"],
    }


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
