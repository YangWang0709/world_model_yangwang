"""Build Step26 BridgeData TFDS world-model smoke samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from data.bridgedata_v2_tfds_importance_manifest import (
    read_importance_manifest_jsonl,
    validate_importance_artifact,
)
from data.bridgedata_v2_tfds_token_artifact_loader import (
    EXPECTED_TOKEN_SHAPES,
    load_step24_token_artifact,
    load_step24_token_manifest,
)

EXPECTED_IMPORTANCE_SHAPE = [16, 392]


def load_bridge_tfds_world_model_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    input_cfg = config["input"]
    max_samples = int(config.get("sample_limits", {}).get("max_samples", 4))
    token_records = load_step24_token_manifest(input_cfg["token_manifest_jsonl"])
    importance_records = read_importance_manifest_jsonl(input_cfg["importance_manifest_jsonl"])
    importance_by_id = {str(record["sample_id"]): record for record in importance_records}

    samples: list[dict[str, Any]] = []
    for token_record in token_records[:max_samples]:
        sample_id = str(token_record["sample_id"])
        if sample_id not in importance_by_id:
            raise FileNotFoundError(f"missing Step25 importance record for sample_id={sample_id}")
        token_sample = load_step24_token_artifact(token_record)
        importance_record = importance_by_id[sample_id]
        importance_artifact = validate_importance_artifact(importance_record["importance_artifact_path"])
        context_importance = importance_artifact["context_importance_norm"].detach().to(
            device="cpu", dtype=torch.float32
        )
        sample = {
            "sample_id": sample_id,
            "trajectory_id": token_sample.get("trajectory_id") or importance_record.get("trajectory_id"),
            "context_tokens": token_sample["context_tokens"],
            "current_tokens": token_sample["current_tokens"],
            "future_tokens": token_sample["future_tokens"],
            "context_importance": context_importance.contiguous(),
            "metadata": {
                "token_artifact_path": token_sample["token_artifact_path"],
                "importance_artifact_path": importance_record["importance_artifact_path"],
                "current_tokens_kept_full": True,
                "train_current_importance": False,
                "action_used_as_input": False,
                "language_used_as_input": False,
                "goal_used_as_input": False,
            },
        }
        validate_world_model_sample(sample)
        samples.append(sample)
    return samples


def validate_world_model_sample(sample: dict[str, Any]) -> bool:
    expected = {
        "context_tokens": EXPECTED_TOKEN_SHAPES["context"],
        "current_tokens": EXPECTED_TOKEN_SHAPES["current"],
        "future_tokens": EXPECTED_TOKEN_SHAPES["future"],
        "context_importance": EXPECTED_IMPORTANCE_SHAPE,
    }
    for key, shape in expected.items():
        value = sample.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{key} must be a tensor")
        if list(value.shape) != shape:
            raise ValueError(f"{key} shape {list(value.shape)} != expected {shape}")
        if value.device.type != "cpu":
            raise ValueError(f"{key} must be CPU for Step26 smoke")
        if bool(value.requires_grad):
            raise ValueError(f"{key} must not require gradients")
    metadata = sample.get("metadata") or {}
    if not bool(metadata.get("current_tokens_kept_full", False)):
        raise ValueError("current_tokens_kept_full must be true")
    if bool(metadata.get("train_current_importance", False)):
        raise ValueError("train_current_importance must be false")
    for flag in ("action_used_as_input", "language_used_as_input", "goal_used_as_input"):
        if bool(metadata.get(flag, False)):
            raise ValueError(f"{flag} must be false")
    return True


def summarize_world_model_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    for sample in samples:
        validate_world_model_sample(sample)
    if not samples:
        return {
            "num_samples": 0,
            "sample_ids": [],
            "token_shapes": {},
            "importance_shape": None,
            "current_tokens_kept_full": True,
            "train_current_importance": False,
        }
    first = samples[0]
    return {
        "num_samples": len(samples),
        "sample_ids": [str(sample["sample_id"]) for sample in samples],
        "token_shapes": {
            "context": list(first["context_tokens"].shape),
            "current": list(first["current_tokens"].shape),
            "future": list(first["future_tokens"].shape),
        },
        "importance_shape": list(first["context_importance"].shape),
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "metadata_only_fields": ["action", "language", "goal"],
    }


def step24_step25_inputs_missing(config: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in (
        "token_manifest_jsonl",
        "token_summary_json",
        "token_smoke_dir",
        "importance_manifest_jsonl",
        "importance_summary_json",
        "importance_smoke_dir",
    ):
        path = Path(config["input"][key])
        if not path.exists():
            missing.append(str(path))
    if not missing:
        token_summary = _read_json(config["input"]["token_summary_json"])
        importance_summary = _read_json(config["input"]["importance_summary_json"])
        if not bool(token_summary.get("token_extraction_performed")):
            missing.append("Step24 token_summary_json reports token_extraction_performed=false")
        if not bool(importance_summary.get("importance_generation_performed")):
            missing.append("Step25 importance_summary_json reports importance_generation_performed=false")
    return missing


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
