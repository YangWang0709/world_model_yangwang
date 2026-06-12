"""Load Step24 BridgeData TFDS token artifacts for Step25 importance dry-runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch

EXPECTED_TOKEN_SHAPES = {
    "context": [16, 392, 768],
    "current": [4, 392, 768],
    "future": [4, 392, 768],
}


def load_step24_token_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read the Step24 JSONL manifest without importing TFDS/TensorFlow."""

    manifest_path = Path(path)
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid token manifest JSON on line {line_number}: {exc}") from exc
            _validate_manifest_record(record)
            records.append(record)
    return records


def load_step24_token_artifact(record: dict[str, Any]) -> dict[str, Any]:
    """Load one Step24 token artifact and return CPU float32 tensors."""

    _validate_manifest_record(record)
    artifact_path = Path(str(record["token_artifact_path"]))
    if not artifact_path.exists():
        raise FileNotFoundError(f"Step24 token artifact is missing: {artifact_path}")
    artifact = _torch_load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError(f"Step24 token artifact must be a dict: {artifact_path}")

    sample = {
        "sample_id": str(artifact.get("sample_id") or record.get("sample_id") or artifact_path.stem),
        "trajectory_id": artifact.get("trajectory_id") or record.get("trajectory_id"),
        "context_tokens": _as_cpu_float32(artifact.get("context_tokens"), "context_tokens"),
        "current_tokens": _as_cpu_float32(artifact.get("current_tokens"), "current_tokens"),
        "future_tokens": _as_cpu_float32(artifact.get("future_tokens"), "future_tokens"),
        "metadata": dict(artifact.get("metadata") or {}),
        "token_artifact_path": str(artifact_path),
        "action_used_as_input": bool(record.get("action_used_as_input", False)),
        "language_used_as_input": bool(record.get("language_used_as_input", False)),
        "goal_used_as_input": bool(record.get("goal_used_as_input", False)),
    }
    validate_step24_token_sample(sample, EXPECTED_TOKEN_SHAPES)
    return sample


def validate_step24_token_sample(sample: dict[str, Any], expected_shapes: dict[str, list[int]]) -> bool:
    required = {
        "context_tokens": expected_shapes["context"],
        "current_tokens": expected_shapes["current"],
        "future_tokens": expected_shapes["future"],
    }
    for key, expected_shape in required.items():
        value = sample.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{key} must be a torch.Tensor")
        if list(value.shape) != list(expected_shape):
            raise ValueError(f"{key} shape {list(value.shape)} != expected {list(expected_shape)}")
        if value.device.type != "cpu":
            raise ValueError(f"{key} must be loaded on CPU")
        if value.dtype != torch.float32:
            raise ValueError(f"{key} must be float32 after loading")
        if bool(value.requires_grad):
            raise ValueError(f"{key} must not require gradients")
    for flag in ("action_used_as_input", "language_used_as_input", "goal_used_as_input"):
        if bool(sample.get(flag)):
            raise ValueError(f"{flag} must be false for Step25 importance")
    return True


def iter_step24_token_samples(manifest_path: str | Path, max_samples: int) -> Iterator[dict[str, Any]]:
    if max_samples < 1:
        return
    for record in load_step24_token_manifest(manifest_path)[:max_samples]:
        yield load_step24_token_artifact(record)


def _validate_manifest_record(record: dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        raise ValueError("token manifest record must be a dict")
    if "token_artifact_path" not in record:
        raise ValueError("token manifest record is missing token_artifact_path")
    expected_pairs = {
        "context_token_shape": EXPECTED_TOKEN_SHAPES["context"],
        "current_token_shape": EXPECTED_TOKEN_SHAPES["current"],
        "future_token_shape": EXPECTED_TOKEN_SHAPES["future"],
    }
    for key, expected_shape in expected_pairs.items():
        shape = record.get(key)
        if list(shape or []) != expected_shape:
            raise ValueError(f"{key} {shape!r} != expected {expected_shape!r}")
    for flag in ("action_used_as_input", "language_used_as_input", "goal_used_as_input"):
        if bool(record.get(flag)):
            raise ValueError(f"{flag} must be false in Step24 token manifest")
    return True


def _as_cpu_float32(value: Any, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} missing or not a tensor")
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
