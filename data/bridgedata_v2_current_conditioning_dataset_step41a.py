"""Dataset helpers for Step41A current-conditioning diagnosis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from data.bridgedata_v2_proxy_patch_selector_splits_step36 import build_step36_training_settings
from data.bridgedata_v2_redesigned_label_selector_dataset_step40a import (
    attach_targets as _attach_step40a_targets,
    batch_step40a_samples,
    build_global_spatial_prior_removed_residual_target,
    build_train_split_global_spatial_prior,
    load_step40a_samples,
    make_step40a_sample_from_tensors,
    validate_step40a_sample,
)


STAGE = "bridgedata_v2_tfds_current_conditioning_diagnosis_step41a"
OPTIMIZER_SCOPE = "current_conditioned_selector_head_only"


def load_step41a_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Step41A config must be a mapping: {path}")
    if config.get("stage") != STAGE:
        raise ValueError(f"unexpected Step41A stage: {config.get('stage')!r}")
    return config


def missing_step41a_inputs(config: dict[str, Any]) -> list[str]:
    input_cfg = config.get("input", {})
    required = [
        input_cfg["step40a_run_dir"],
        input_cfg["step40a_train_summary_json"],
        input_cfg["step40a_selector_metrics_json"],
        input_cfg["step40a_gate_decision_json"],
        input_cfg["step39a_run_dir"],
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

    step40a_summary = _read_json(input_cfg["step40a_train_summary_json"])
    step40a_gate = _read_json(input_cfg["step40a_gate_decision_json"])
    step39a_gate = _read_json(input_cfg["step39a_gate_decision_json"])
    step37_gate = _read_json(input_cfg["step37_gate_decision_json"])
    if bool(step40a_summary.get("safe_stop", False)):
        missing.append("Step40A summary safe_stop is true")
    if not bool(step40a_summary.get("bounded_selector_smoke_training_performed", False)):
        missing.append("Step40A bounded selector smoke was not performed")
    if bool(step40a_gate.get("downstream_selector_use_allowed", False)):
        missing.append("unsafe upstream Step40A gate enabled downstream selector use")
    if bool(step40a_gate.get("redesigned_label_selector_smoke_pass", True)):
        missing.append("Step40A unexpectedly passed; Step41A diagnosis expects the documented failed Step40A gate")
    for label, gate in (
        ("Step40A", step40a_gate),
        ("Step39A", step39a_gate),
        ("Step37", step37_gate),
    ):
        for flag in (
            "final_selector_training_allowed",
            "current_importance_training_allowed",
            "context_utility_claim_allowed",
        ):
            if bool(gate.get(flag, False)):
                missing.append(f"unsafe upstream {label} gate flag is true: {flag}")
    return missing


def load_step41a_samples(config: dict[str, Any]) -> dict[str, Any]:
    step40a_like = _step40a_compatible_config(config)
    payload = load_step40a_samples(step40a_like)
    payload["stage"] = STAGE
    payload["summary"]["stage"] = STAGE
    payload["summary"]["future_tokens_exposed_to_selector"] = False
    payload["summary"]["action_used_as_input"] = False
    payload["summary"]["language_used_as_input"] = False
    payload["summary"]["current_importance_training_performed"] = False
    return payload


def build_step41a_training_settings(config: dict[str, Any]) -> list[dict[str, Any]]:
    step36_like = _step40a_compatible_config(config)
    step36_like["stage"] = "bridgedata_v2_tfds_proxy_patch_selector_train_step36"
    settings = build_step36_training_settings(step36_like)
    for setting in settings:
        setting["stage"] = STAGE
    return settings


def build_step41a_training_settings_for_fake(
    *,
    split_seed: int = 42,
    run_within_shard: bool = True,
    run_cross_shard: bool = True,
    run_mixed_shard: bool = True,
) -> list[dict[str, Any]]:
    settings = []
    if run_within_shard:
        settings.append(
            _setting(
                "within_shard_seed42",
                "within_shard",
                split_seed,
                {"shard1": ["s0", "s1"]},
                {"shard1": ["s2"]},
            )
        )
    if run_cross_shard:
        settings.append(
            _setting(
                "cross_shard_train0_val1_seed42",
                "cross_shard",
                split_seed,
                {"shard0": ["z0", "z1"]},
                {"shard1": ["s2"]},
            )
        )
    if run_mixed_shard:
        settings.append(
            _setting(
                "mixed_shard_seed42",
                "mixed_shard",
                split_seed,
                {"shard0": ["z0"], "shard1": ["s0"]},
                {"shard0": ["z1"], "shard1": ["s2"]},
            )
        )
    return settings


def make_step41a_sample_from_tensors(
    *,
    sample_id: str,
    trajectory_id: str,
    shard_id: str,
    data_package_id: str,
    context_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    original_importance: torch.Tensor,
) -> dict[str, Any]:
    return make_step40a_sample_from_tensors(
        sample_id=sample_id,
        trajectory_id=trajectory_id,
        shard_id=shard_id,
        data_package_id=data_package_id,
        context_tokens=context_tokens,
        current_tokens=current_tokens,
        original_importance=original_importance,
    )


def validate_step41a_sample(sample: dict[str, Any]) -> bool:
    return validate_step40a_sample(sample)


def build_step41a_target_label(importance: torch.Tensor, train_prior: torch.Tensor) -> torch.Tensor:
    return build_global_spatial_prior_removed_residual_target(importance, train_prior)


def attach_step41a_targets(samples: list[dict[str, Any]], train_prior: torch.Tensor) -> list[dict[str, Any]]:
    return _attach_step40a_targets(samples, train_prior)


def batch_step41a_samples(samples: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    batch = batch_step40a_samples(samples, device)
    for key in ("context_tokens", "current_tokens", "target_label"):
        if batch[key].dtype != torch.float32:
            raise ValueError(f"{key} must be float32")
        if bool(batch[key].requires_grad):
            raise ValueError(f"{key} must not require gradients")
    return batch


def _step40a_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "bridgedata_v2_tfds_redesigned_label_selector_step40a",
        "project_root": config.get("project_root"),
        "seed": config.get("seed", 42),
        "input": {
            "step39a_run_dir": config["input"]["step39a_run_dir"],
            "step39a_gate_decision_json": config["input"]["step39a_gate_decision_json"],
            "step37_run_dir": config["input"]["step37_run_dir"],
            "step37_gate_decision_json": config["input"]["step37_gate_decision_json"],
            "token_manifest_jsonl": config["input"]["token_manifest_jsonl"],
            "importance_manifest_jsonl": config["input"]["importance_manifest_jsonl"],
            "shard_splits_json": config["input"]["shard_splits_json"],
            "shard0_token_manifest_jsonl": config["input"]["shard0_token_manifest_jsonl"],
            "shard0_importance_manifest_jsonl": config["input"]["shard0_importance_manifest_jsonl"],
            "shard0_horizon_window_manifest_jsonl": config["input"]["shard0_horizon_window_manifest_jsonl"],
            "shard0_horizon_splits_json": config["input"]["shard0_horizon_splits_json"],
        },
        "data": config.get("data", {}),
        "label": config.get("label", {}),
        "splits": config.get("splits", {}),
    }


def _setting(
    name: str,
    eval_type: str,
    split_seed: int,
    train: dict[str, list[str]],
    val: dict[str, list[str]],
) -> dict[str, Any]:
    train_ids = {sample_id for ids in train.values() for sample_id in ids}
    val_ids = {sample_id for ids in val.values() for sample_id in ids}
    return {
        "stage": STAGE,
        "name": name,
        "eval_type": eval_type,
        "split_seed": int(split_seed),
        "train_sample_ids_by_shard": train,
        "val_sample_ids_by_shard": val,
        "train_trajectories_by_shard": {},
        "val_trajectories_by_shard": {},
        "num_train_samples": len(train_ids),
        "num_val_samples": len(val_ids),
        "train_val_sample_id_disjoint": not bool(train_ids & val_ids),
        "train_val_trajectory_disjoint_or_documented": True,
        "strict_shard_aware_split": True,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
