"""Dataset and redesigned-label helpers for Step40A selector smoke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from data.bridgedata_v2_proxy_label_redesign_step39a import (
    build_global_spatial_prior,
    build_global_spatial_prior_removed_residual_label,
)
from data.bridgedata_v2_proxy_patch_selector_splits_step36 import build_step36_training_settings
from data.bridgedata_v2_tfds_importance_manifest import read_importance_manifest_jsonl, validate_importance_artifact
from data.bridgedata_v2_tfds_longer_horizon_manifest import read_jsonl
from data.bridgedata_v2_tfds_token_artifact_loader import load_step24_token_manifest


STAGE = "bridgedata_v2_tfds_redesigned_label_selector_step40a"
OPTIMIZER_SCOPE = "redesigned_label_proxy_selector_head_only"


def load_step40a_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Step40A config must be a mapping: {path}")
    if config.get("stage") != STAGE:
        raise ValueError(f"unexpected Step40A stage: {config.get('stage')!r}")
    return config


def missing_step40a_inputs(config: dict[str, Any]) -> list[str]:
    input_cfg = config.get("input", {})
    required = [
        input_cfg["step39a_run_dir"],
        input_cfg["step39a_label_redesign_summary_json"],
        input_cfg["step39a_label_variant_metrics_json"],
        input_cfg["step39a_label_variant_comparison_json"],
        input_cfg["step39a_gate_decision_json"],
        input_cfg["step37_run_dir"],
        input_cfg["step37_gate_decision_json"],
        input_cfg["token_manifest_jsonl"],
        input_cfg["importance_manifest_jsonl"],
        input_cfg["shard_splits_json"],
        input_cfg["shard0_token_manifest_jsonl"],
        input_cfg["shard0_importance_manifest_jsonl"],
        input_cfg["shard0_horizon_window_manifest_jsonl"],
        input_cfg["shard0_horizon_splits_json"],
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        return missing
    step39a_gate = _read_json(input_cfg["step39a_gate_decision_json"])
    if step39a_gate.get("recommended_step40_label_variant") != config["label"]["variant"]:
        missing.append("Step39A recommended label variant does not match Step40A config")
    if not bool(step39a_gate.get("redesigned_label_candidate_ready", False)):
        missing.append("Step39A redesigned_label_candidate_ready is false")
    for label, gate in (
        ("Step39A", step39a_gate),
        ("Step37", _read_json(input_cfg["step37_gate_decision_json"])),
    ):
        for flag in ("final_selector_training_allowed", "current_importance_training_allowed", "context_utility_claim_allowed"):
            if bool(gate.get(flag, False)):
                missing.append(f"unsafe upstream {label} gate flag is true: {flag}")
    return missing


def build_step40a_training_settings(config: dict[str, Any]) -> list[dict[str, Any]]:
    step36_like = dict(config)
    step36_like["stage"] = "bridgedata_v2_tfds_proxy_patch_selector_train_step36"
    return build_step36_training_settings(step36_like)


def sample_ids_by_shard_from_splits(config: dict[str, Any]) -> dict[str, set[str]]:
    ids = {"shard0": set(), "shard1": set()}
    for setting in build_step40a_training_settings(config):
        for split_key in ("train_sample_ids_by_shard", "val_sample_ids_by_shard"):
            for shard, sample_ids in setting[split_key].items():
                ids.setdefault(str(shard), set()).update(str(sample_id) for sample_id in sample_ids)
    return ids


def load_step40a_samples(config: dict[str, Any]) -> dict[str, Any]:
    required_ids = sample_ids_by_shard_from_splits(config)
    max_per_shard = int(config.get("data", {}).get("max_samples_per_shard", 64))
    shard1 = _load_samples_for_manifest(
        token_manifest_jsonl=config["input"]["token_manifest_jsonl"],
        importance_manifest_jsonl=config["input"]["importance_manifest_jsonl"],
        allowed_sample_ids=set(required_ids.get("shard1", set())),
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
    shard0 = _load_samples_for_manifest(
        token_manifest_jsonl=config["input"]["shard0_token_manifest_jsonl"],
        importance_manifest_jsonl=config["input"]["shard0_importance_manifest_jsonl"],
        allowed_sample_ids=set(required_ids.get("shard0", set())) & shard0_gap0_ids,
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
            "label_variant": config["label"]["variant"],
            "target_label_shape": list(config["data"].get("importance_shape", [16, 392])),
        },
    }


def make_step40a_sample_from_tensors(
    *,
    sample_id: str,
    trajectory_id: str,
    shard_id: str,
    data_package_id: str,
    context_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    original_importance: torch.Tensor,
) -> dict[str, Any]:
    sample = {
        "sample_id": str(sample_id),
        "trajectory_id": str(trajectory_id),
        "shard_id": str(shard_id),
        "data_package_id": str(data_package_id),
        "context_tokens": _tensor(context_tokens, "context_tokens"),
        "current_tokens": _tensor(current_tokens, "current_tokens"),
        "original_importance": _tensor(original_importance, "original_importance"),
        "metadata": {
            "future_tokens_exposed_to_selector": False,
            "action_used_as_input": False,
            "language_used_as_input": False,
            "goal_used_as_input": False,
            "current_importance_generated": False,
            "train_current_importance": False,
        },
    }
    validate_step40a_sample(sample)
    return sample


def validate_step40a_sample(sample: dict[str, Any]) -> bool:
    context = sample.get("context_tokens")
    current = sample.get("current_tokens")
    importance = sample.get("original_importance")
    if not isinstance(context, torch.Tensor) or context.ndim != 3:
        raise ValueError("context_tokens must be [T,S,D]")
    if not isinstance(current, torch.Tensor) or current.ndim != 3:
        raise ValueError("current_tokens must be [Tcur,S,D]")
    if not isinstance(importance, torch.Tensor) or importance.ndim != 2:
        raise ValueError("original_importance must be [T,S]")
    if list(importance.shape) != [int(context.shape[0]), int(context.shape[1])]:
        raise ValueError("original_importance shape must match context temporal/spatial axes")
    for key, tensor in {"context_tokens": context, "current_tokens": current, "original_importance": importance}.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"{key} must be float32")
        if tensor.device.type != "cpu":
            raise ValueError(f"{key} must be CPU")
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


def build_train_split_global_spatial_prior(train_samples: list[dict[str, Any]]) -> dict[str, Any]:
    payload = build_global_spatial_prior([{"importance": sample["original_importance"]} for sample in train_samples])
    return {
        "prior": payload["prior"].detach().to(dtype=torch.float32, device="cpu").contiguous(),
        "stats": {
            **payload["stats"],
            "built_from_train_split_only": True,
            "val_split_used_for_prior": False,
            "num_train_samples": len(train_samples),
        },
    }


def build_global_spatial_prior_removed_residual_target(importance: torch.Tensor, train_prior: torch.Tensor) -> torch.Tensor:
    return build_global_spatial_prior_removed_residual_label(importance, train_prior).detach().clamp(0.0, 1.0).to(
        dtype=torch.float32,
        device="cpu",
    )


def attach_targets(samples: list[dict[str, Any]], train_prior: torch.Tensor) -> list[dict[str, Any]]:
    result = []
    for sample in samples:
        clone = dict(sample)
        clone["target_label"] = build_global_spatial_prior_removed_residual_target(sample["original_importance"], train_prior)
        result.append(clone)
    return result


def batch_step40a_samples(samples: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    for sample in samples:
        validate_step40a_sample(sample)
        target = sample.get("target_label")
        if not isinstance(target, torch.Tensor) or target.ndim != 2:
            raise ValueError("target_label must be [T,S]")
        if target.dtype != torch.float32 or target.device.type != "cpu" or bool(target.requires_grad):
            raise ValueError("target_label must be detached CPU float32")
    return {
        "context_tokens": torch.stack([sample["context_tokens"] for sample in samples]).to(device),
        "current_tokens": torch.stack([sample["current_tokens"] for sample in samples]).to(device),
        "target_label": torch.stack([sample["target_label"] for sample in samples]).to(device),
    }


def _load_samples_for_manifest(
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
    samples = []
    for sample_id in ordered_ids[: int(max_samples)]:
        if sample_id not in importance_by_id:
            raise FileNotFoundError(f"missing importance record for sample_id={sample_id}")
        samples.append(
            _load_sample(
                token_record=token_by_id[sample_id],
                importance_record=importance_by_id[sample_id],
                shard_id=shard_id,
                data_package_id=data_package_id,
                expected=expected,
            )
        )
    return samples


def _load_sample(
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
    if list(context_tokens.shape) != list(expected.get("context_token_shape", [16, 392, 768])):
        raise ValueError("context_tokens shape does not match Step40A config")
    if list(current_tokens.shape) != list(expected.get("current_token_shape", [4, 392, 768])):
        raise ValueError("current_tokens shape does not match Step40A config")
    importance_artifact = validate_importance_artifact(importance_record["importance_artifact_path"])
    sample = make_step40a_sample_from_tensors(
        sample_id=str(token_record["sample_id"]),
        trajectory_id=str(token_record.get("trajectory_id") or importance_record.get("trajectory_id")),
        shard_id=shard_id,
        data_package_id=data_package_id,
        context_tokens=context_tokens,
        current_tokens=current_tokens,
        original_importance=importance_artifact["context_importance_norm"],
    )
    sample["metadata"].update(
        {
            "token_artifact_path": str(token_record["token_artifact_path"]),
            "importance_artifact_path": str(importance_record["importance_artifact_path"]),
        }
    )
    return sample


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
