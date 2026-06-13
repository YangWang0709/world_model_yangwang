"""Dataset helpers for Step37 factorized selector diagnosis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from data.bridgedata_v2_proxy_patch_selector_splits_step36 import build_step36_training_settings
from data.bridgedata_v2_tfds_importance_manifest import (
    read_importance_manifest_jsonl,
    validate_importance_artifact,
)
from data.bridgedata_v2_tfds_longer_horizon_manifest import read_jsonl
from data.bridgedata_v2_tfds_token_artifact_loader import load_step24_token_manifest
from models.bridgedata_v2_proxy_factorized_selector import proxy_patch_and_temporal_targets


STAGE = "bridgedata_v2_tfds_factorized_selector_step37"
OPTIMIZER_SCOPE = "proxy_factorized_selector_head_only"
ABLATION_OPTIMIZER_SCOPE = "proxy_selector_ablation_heads_only"


def load_step37_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Step37 config must be a mapping: {path}")
    if config.get("stage") != STAGE:
        raise ValueError(f"unexpected Step37 stage: {config.get('stage')!r}")
    return config


def missing_step37_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["input"]["token_manifest_jsonl"],
        config["input"]["token_summary_json"],
        config["input"]["importance_manifest_jsonl"],
        config["input"]["importance_summary_json"],
        config["input"]["shard_splits_json"],
        config["input"]["step35_selector_metrics_json"],
        config["input"]["step35_gate_decision_json"],
        config["input"]["step36_patch_selector_metrics_json"],
        config["input"]["step36_gate_decision_json"],
        config["input"]["step36_eval_json"],
        config["input"]["shard0_token_manifest_jsonl"],
        config["input"]["shard0_token_summary_json"],
        config["input"]["shard0_importance_manifest_jsonl"],
        config["input"]["shard0_importance_summary_json"],
        config["input"]["shard0_horizon_window_manifest_jsonl"],
        config["input"]["shard0_horizon_splits_json"],
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if not missing:
        step36_eval = _read_json(config["input"]["step36_eval_json"])
        if not bool(step36_eval.get("pass")):
            missing.append("Step36 eval did not pass")
    return missing


def build_step37_training_settings(config: dict[str, Any]) -> list[dict[str, Any]]:
    step36_like = dict(config)
    step36_like["stage"] = "bridgedata_v2_tfds_proxy_patch_selector_train_step36"
    return build_step36_training_settings(step36_like)


def sample_ids_by_shard_from_splits(config: dict[str, Any]) -> dict[str, set[str]]:
    settings = build_step37_training_settings(config)
    ids = {"shard0": set(), "shard1": set()}
    for setting in settings:
        for split_key in ("train_sample_ids_by_shard", "val_sample_ids_by_shard"):
            for shard, sample_ids in setting[split_key].items():
                ids.setdefault(str(shard), set()).update(str(sample_id) for sample_id in sample_ids)
    return ids


def load_step37_samples(config: dict[str, Any], required_ids_by_shard: dict[str, set[str]]) -> dict[str, Any]:
    max_per_shard = int(config.get("data", {}).get("max_samples_per_shard", 64))
    shard1 = load_factorized_selector_samples(
        token_manifest_jsonl=config["input"]["token_manifest_jsonl"],
        importance_manifest_jsonl=config["input"]["importance_manifest_jsonl"],
        allowed_sample_ids=set(required_ids_by_shard.get("shard1", set())),
        shard_id="shard1",
        data_package_id="bridge_tfds_shard1",
        expected=config["data"],
        max_samples=max_per_shard,
    )
    shard0_gap0_ids = {
        str(record["sample_id"])
        for record in read_jsonl(config["input"]["shard0_horizon_window_manifest_jsonl"])
        if int(record.get("horizon_gap", -1)) == 0
    }
    shard0 = load_factorized_selector_samples(
        token_manifest_jsonl=config["input"]["shard0_token_manifest_jsonl"],
        importance_manifest_jsonl=config["input"]["shard0_importance_manifest_jsonl"],
        allowed_sample_ids=set(required_ids_by_shard.get("shard0", set())) & shard0_gap0_ids,
        shard_id="shard0",
        data_package_id="bridge_tfds_shard0_gap0",
        expected=config["data"],
        max_samples=max_per_shard,
    )
    return {
        "stage": STAGE,
        "samples_by_shard": {"shard0": shard0, "shard1": shard1},
        "summary": {
            "shard0_num_samples": len(shard0),
            "shard1_num_samples": len(shard1),
            "future_tokens_exposed_to_selector": False,
            "action_used_as_input": False,
            "language_used_as_input": False,
            "current_importance_generated": False,
            "proxy_patch_target_shape": [16, 392],
            "proxy_temporal_target_shape": [16],
        },
    }


def load_factorized_selector_samples(
    *,
    token_manifest_jsonl: str | Path,
    importance_manifest_jsonl: str | Path,
    allowed_sample_ids: set[str],
    shard_id: str,
    data_package_id: str,
    expected: dict[str, Any],
    max_samples: int,
) -> list[dict[str, Any]]:
    token_records = load_step24_token_manifest(token_manifest_jsonl)
    importance_records = read_importance_manifest_jsonl(importance_manifest_jsonl)
    token_by_id = {str(record["sample_id"]): record for record in token_records}
    importance_by_id = {str(record["sample_id"]): record for record in importance_records}
    ordered_ids = [str(record["sample_id"]) for record in token_records if str(record["sample_id"]) in allowed_sample_ids]
    samples: list[dict[str, Any]] = []
    for sample_id in ordered_ids[: int(max_samples)]:
        if sample_id not in importance_by_id:
            raise FileNotFoundError(f"missing importance record for sample_id={sample_id}")
        samples.append(
            load_factorized_selector_sample(
                token_record=token_by_id[sample_id],
                importance_record=importance_by_id[sample_id],
                shard_id=shard_id,
                data_package_id=data_package_id,
                expected=expected,
            )
        )
    return samples


def load_factorized_selector_sample(
    *,
    token_record: dict[str, Any],
    importance_record: dict[str, Any],
    shard_id: str,
    data_package_id: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    token_artifact = _torch_load(Path(str(token_record["token_artifact_path"])))
    if not isinstance(token_artifact, dict):
        raise ValueError("token artifact must be a dict")
    context_tokens = _tensor(token_artifact.get("context_tokens"), "context_tokens")
    current_tokens = _tensor(token_artifact.get("current_tokens"), "current_tokens")
    context_shape = list(expected.get("context_token_shape", [16, 392, 768]))
    current_shape = list(expected.get("current_token_shape", [4, 392, 768]))
    if list(context_tokens.shape) != context_shape:
        raise ValueError(f"context_tokens shape {list(context_tokens.shape)} != {context_shape}")
    if list(current_tokens.shape) != current_shape:
        raise ValueError(f"current_tokens shape {list(current_tokens.shape)} != {current_shape}")
    importance_artifact = validate_importance_artifact(importance_record["importance_artifact_path"])
    targets = proxy_patch_and_temporal_targets(importance_artifact["context_importance_norm"])
    sample = make_step37_sample_from_tensors(
        sample_id=str(token_record["sample_id"]),
        trajectory_id=str(token_record.get("trajectory_id") or importance_record.get("trajectory_id")),
        shard_id=shard_id,
        data_package_id=data_package_id,
        context_tokens=context_tokens,
        current_tokens=current_tokens,
        proxy_patch_target=targets["proxy_patch_target"].squeeze(0),
        proxy_temporal_target=targets["proxy_temporal_target"].squeeze(0),
    )
    sample["metadata"].update(
        {
            "token_artifact_path": str(token_record["token_artifact_path"]),
            "importance_artifact_path": str(importance_record["importance_artifact_path"]),
        }
    )
    return sample


def make_step37_sample_from_tensors(
    *,
    sample_id: str,
    trajectory_id: str,
    shard_id: str,
    data_package_id: str,
    context_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    proxy_patch_target: torch.Tensor,
    proxy_temporal_target: torch.Tensor,
) -> dict[str, Any]:
    context = context_tokens.detach().to(dtype=torch.float32, device="cpu").contiguous()
    current = current_tokens.detach().to(dtype=torch.float32, device="cpu").contiguous()
    patch = proxy_patch_target.detach().to(dtype=torch.float32, device="cpu").contiguous()
    temporal = proxy_temporal_target.detach().to(dtype=torch.float32, device="cpu").reshape(-1).contiguous()
    if context.ndim != 3:
        raise ValueError(f"context_tokens must be [T, S, D], got {tuple(context.shape)}")
    if current.ndim != 3:
        raise ValueError(f"current_tokens must be [Tcur, S, D], got {tuple(current.shape)}")
    if patch.ndim != 2 or list(patch.shape) != [int(context.shape[0]), int(context.shape[1])]:
        raise ValueError("proxy_patch_target must be [T, S] matching context tokens")
    if int(temporal.numel()) != int(context.shape[0]):
        raise ValueError("proxy_temporal_target must have one score per context frame")
    sample = {
        "sample_id": str(sample_id),
        "trajectory_id": str(trajectory_id),
        "shard_id": str(shard_id),
        "data_package_id": str(data_package_id),
        "context_tokens": context,
        "current_tokens": current,
        "proxy_patch_target": patch,
        "proxy_temporal_target": temporal,
        "metadata": {
            "future_tokens_exposed_to_selector": False,
            "action_used_as_input": False,
            "language_used_as_input": False,
            "goal_used_as_input": False,
            "current_importance_generated": False,
            "train_current_importance": False,
        },
    }
    validate_step37_sample(sample)
    return sample


def validate_step37_sample(sample: dict[str, Any]) -> bool:
    context = sample.get("context_tokens")
    current = sample.get("current_tokens")
    patch = sample.get("proxy_patch_target")
    temporal = sample.get("proxy_temporal_target")
    if not isinstance(context, torch.Tensor) or context.ndim != 3:
        raise ValueError("context_tokens must be [T, S, D]")
    if not isinstance(current, torch.Tensor) or current.ndim != 3:
        raise ValueError("current_tokens must be [Tcur, S, D]")
    if not isinstance(patch, torch.Tensor) or patch.ndim != 2:
        raise ValueError("proxy_patch_target must be [T, S]")
    if not isinstance(temporal, torch.Tensor) or temporal.ndim != 1:
        raise ValueError("proxy_temporal_target must be [T]")
    if list(patch.shape) != [int(context.shape[0]), int(context.shape[1])]:
        raise ValueError("proxy_patch_target shape must match context temporal/spatial axes")
    if int(temporal.shape[0]) != int(context.shape[0]):
        raise ValueError("proxy_temporal_target length must match context frames")
    for key, tensor in {
        "context_tokens": context,
        "current_tokens": current,
        "proxy_patch_target": patch,
        "proxy_temporal_target": temporal,
    }.items():
        if tensor.device.type != "cpu":
            raise ValueError(f"{key} must be CPU")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{key} must be float32")
        if bool(tensor.requires_grad):
            raise ValueError(f"{key} must not require gradients")
    metadata = sample.get("metadata") or {}
    for flag in (
        "future_tokens_exposed_to_selector",
        "action_used_as_input",
        "language_used_as_input",
        "goal_used_as_input",
        "current_importance_generated",
        "train_current_importance",
    ):
        if bool(metadata.get(flag, False)):
            raise ValueError(f"{flag} must be false")
    return True


def batch_step37_samples(samples: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    for sample in samples:
        validate_step37_sample(sample)
    return {
        "context_tokens": torch.stack([sample["context_tokens"] for sample in samples]).to(device),
        "current_tokens": torch.stack([sample["current_tokens"] for sample in samples]).to(device),
        "proxy_patch_target": torch.stack([sample["proxy_patch_target"] for sample in samples]).to(device),
        "proxy_temporal_target": torch.stack([sample["proxy_temporal_target"] for sample in samples]).to(device),
    }


def _tensor(value: Any, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} missing or not tensor")
    return value.detach().to(dtype=torch.float32, device="cpu").contiguous()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
