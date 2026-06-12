"""Shape-flexible token schema for Step33A true-temporal BridgeData diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


STAGE = "bridgedata_v2_tfds_true_temporal_step33a"
TOKENIZATION_MODE = "true_temporal_clip"
TOKEN_DIM = 768
DEFAULT_REQUIRED_FRAMES = 16
METHOD_TARGETS = (
    "future_delta_last_minus_current",
    "future_delta_mean_minus_current",
    "future_mean_all4",
)


def read_true_temporal_manifest(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid Step33A token manifest line {line_number}: {exc}") from exc
            validate_true_temporal_manifest_record(record)
            records.append(record)
    return records


def write_true_temporal_manifest(records: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_true_temporal_manifest_record(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return output


def validate_true_temporal_manifest_record(record: dict[str, Any]) -> bool:
    required = (
        "sample_id",
        "trajectory_id",
        "horizon_gap",
        "tokenization_mode",
        "token_artifact_path",
        "context_token_shape",
        "current_token_shape",
        "future_token_shape",
        "raw_token_shapes",
    )
    for key in required:
        if key not in record:
            raise ValueError(f"Step33A token manifest missing {key}")
    if int(record["horizon_gap"]) != 0:
        raise ValueError("Step33A only accepts gap0 windows")
    if str(record["tokenization_mode"]) != TOKENIZATION_MODE:
        raise ValueError(f"unexpected tokenization_mode {record['tokenization_mode']!r}")
    for key in ("context_token_shape", "current_token_shape", "future_token_shape"):
        _validate_flat_token_shape(record[key], key)
    for flag in ("action_used_as_input", "language_used_as_input", "goal_used_as_input"):
        if bool(record.get(flag, False)):
            raise ValueError(f"{flag} must be false for Step33A")
    return True


def load_true_temporal_token_artifact(record: dict[str, Any]) -> dict[str, Any]:
    validate_true_temporal_manifest_record(record)
    artifact_path = Path(str(record["token_artifact_path"]))
    if not artifact_path.exists():
        raise FileNotFoundError(f"Step33A token artifact is missing: {artifact_path}")
    artifact = _torch_load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError(f"Step33A token artifact must be a dict: {artifact_path}")
    sample = {
        "sample_id": str(artifact.get("sample_id") or record["sample_id"]),
        "trajectory_id": artifact.get("trajectory_id") or record.get("trajectory_id"),
        "horizon_gap": int(artifact.get("horizon_gap", record.get("horizon_gap", 0))),
        "tokenization_mode": artifact.get("tokenization_mode") or record.get("tokenization_mode"),
        "context_tokens": _as_flat_tokens(artifact.get("context_tokens"), "context_tokens"),
        "current_tokens": _as_flat_tokens(artifact.get("current_tokens"), "current_tokens"),
        "future_tokens": _as_flat_tokens(artifact.get("future_tokens"), "future_tokens"),
        "metadata": dict(artifact.get("metadata") or {}),
        "raw_token_shapes": dict(artifact.get("raw_token_shapes") or record.get("raw_token_shapes") or {}),
        "token_artifact_path": str(artifact_path),
    }
    sample["context_summary"] = summarize_tokens(sample["context_tokens"])
    sample["current_summary"] = summarize_tokens(sample["current_tokens"])
    sample["future_summary"] = summarize_tokens(sample["future_tokens"])
    sample["future_delta_last_minus_current"] = sample["future_summary"] - sample["current_summary"]
    sample["future_delta_mean_minus_current"] = sample["future_summary"] - sample["current_summary"]
    validate_true_temporal_sample(sample)
    return sample


def validate_true_temporal_sample(sample: dict[str, Any]) -> bool:
    if int(sample.get("horizon_gap", 0)) != 0:
        raise ValueError("Step33A sample must have horizon_gap=0")
    if str(sample.get("tokenization_mode")) != TOKENIZATION_MODE:
        raise ValueError(f"unexpected tokenization_mode {sample.get('tokenization_mode')!r}")
    for key in ("context_tokens", "current_tokens", "future_tokens"):
        _as_flat_tokens(sample.get(key), key)
    for key in ("context_summary", "current_summary", "future_summary"):
        value = sample.get(key)
        if not isinstance(value, torch.Tensor) or list(value.shape) != [TOKEN_DIM]:
            raise ValueError(f"{key} must be a [{TOKEN_DIM}] tensor")
    for key in ("future_delta_last_minus_current", "future_delta_mean_minus_current"):
        value = sample.get(key)
        if not isinstance(value, torch.Tensor) or list(value.shape) != [TOKEN_DIM]:
            raise ValueError(f"{key} must be a [{TOKEN_DIM}] tensor")
    metadata = sample.get("metadata") or {}
    for flag in ("action_used_as_input", "language_used_as_input", "goal_used_as_input"):
        if bool(metadata.get(flag, sample.get(flag, False))):
            raise ValueError(f"{flag} must be false for Step33A")
    return True


def summarize_tokens(tokens: torch.Tensor) -> torch.Tensor:
    tokens = _as_flat_tokens(tokens, "tokens")
    return tokens.mean(dim=0).detach().to(dtype=torch.float32, device="cpu").contiguous()


def target_summary_for_variant(sample: dict[str, Any], target_variant: str) -> torch.Tensor:
    validate_true_temporal_sample(sample)
    if target_variant == "future_delta_last_minus_current":
        return sample["future_delta_last_minus_current"].detach().to(dtype=torch.float32)
    if target_variant == "future_delta_mean_minus_current":
        return sample["future_delta_mean_minus_current"].detach().to(dtype=torch.float32)
    if target_variant == "future_mean_all4":
        return sample["future_summary"].detach().to(dtype=torch.float32)
    raise ValueError(f"unknown Step33A target variant: {target_variant}")


def pack_temporal_clip(video: torch.Tensor, required_frames: int = DEFAULT_REQUIRED_FRAMES) -> torch.Tensor:
    """Return [required_frames, C, H, W] using deterministic order-preserving repeat."""

    if not isinstance(video, torch.Tensor):
        raise ValueError("video must be a torch.Tensor")
    if video.ndim != 4:
        raise ValueError(f"video must have shape [T,C,H,W], got {list(video.shape)}")
    frames = int(video.shape[0])
    if frames <= 0:
        raise ValueError("video must contain at least one frame")
    if frames == int(required_frames):
        return video.detach().to(dtype=torch.float32).contiguous()
    indices = torch.div(
        torch.arange(int(required_frames), dtype=torch.long) * frames,
        int(required_frames),
        rounding_mode="floor",
    ).clamp(max=frames - 1)
    return video.index_select(0, indices).detach().to(dtype=torch.float32).contiguous()


def select_true_temporal_context_tokens(sample: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    validate_true_temporal_sample(sample)
    policy_name = str(policy["name"])
    context_tokens = _as_flat_tokens(sample["context_tokens"], "context_tokens")
    importance = sample.get("context_importance")
    if importance is None:
        importance = torch.ones((context_tokens.shape[0],), dtype=torch.float32)
    importance = _as_importance(importance, int(context_tokens.shape[0]))

    if policy_name == "current_only" or not bool(policy.get("use_context", True)):
        indices = torch.empty((0,), dtype=torch.long)
    elif str(policy.get("selection")) == "random":
        topk = _bounded_topk(policy.get("topk"), int(context_tokens.shape[0]))
        generator = torch.Generator().manual_seed(int(policy.get("seed", 42)))
        indices = torch.randperm(int(context_tokens.shape[0]), generator=generator)[:topk].to(dtype=torch.long)
    elif str(policy.get("selection")) == "importance_topk":
        topk = _bounded_topk(policy.get("topk"), int(context_tokens.shape[0]))
        indices = torch.topk(importance, k=topk, largest=True, sorted=True).indices.to(dtype=torch.long)
    elif str(policy.get("selection")) == "full":
        indices = torch.arange(int(context_tokens.shape[0]), dtype=torch.long)
    else:
        raise ValueError(f"unsupported Step33A policy: {policy!r}")

    selected_tokens = (
        context_tokens.index_select(0, indices) if indices.numel() else context_tokens.new_zeros((0, TOKEN_DIM))
    )
    selected_values = importance.index_select(0, indices) if indices.numel() else importance.new_zeros((0,))
    total_mass = float(importance.sum().item())
    selected_mass = float(selected_values.sum().item())
    return {
        "policy_name": policy_name,
        "selected_context_tokens": selected_tokens.contiguous(),
        "selected_indices": indices.contiguous(),
        "selected_importance_values": selected_values.contiguous(),
        "selected_importance_mass": selected_mass / total_mass if total_mass > 0 else 0.0,
        "selected_importance_sum": selected_mass,
        "selected_importance_mean": float(selected_values.mean().item()) if selected_values.numel() else 0.0,
        "topk": int(indices.numel()) if policy.get("topk") is not None else None,
        "num_selected": int(indices.numel()),
        "current_tokens_kept_full": True,
        "deployable": bool(policy.get("deployable", True)),
    }


def summarize_true_temporal_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_samples": 0,
            "context_token_shape_example": None,
            "current_token_shape_example": None,
            "future_token_shape_example": None,
        }
    for record in records:
        validate_true_temporal_manifest_record(record)
    first = records[0]
    return {
        "num_samples": len(records),
        "num_token_artifacts": len({str(record["token_artifact_path"]) for record in records}),
        "context_token_shape_example": first["context_token_shape"],
        "current_token_shape_example": first["current_token_shape"],
        "future_token_shape_example": first["future_token_shape"],
        "raw_token_shapes": first["raw_token_shapes"],
        "summary_shapes": {
            "context_summary": [TOKEN_DIM],
            "current_summary": [TOKEN_DIM],
            "future_summary": [TOKEN_DIM],
        },
    }


def _as_flat_tokens(value: Any, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} must be a torch.Tensor")
    tokens = value.detach().to(dtype=torch.float32, device="cpu")
    if tokens.ndim != 2 or int(tokens.shape[-1]) != TOKEN_DIM or int(tokens.shape[0]) <= 0:
        raise ValueError(f"{key} must have shape [N,{TOKEN_DIM}], got {list(tokens.shape)}")
    if bool(tokens.requires_grad):
        raise ValueError(f"{key} must not require gradients")
    return tokens.contiguous()


def _validate_flat_token_shape(shape: Any, key: str) -> None:
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"{key} must be a rank-2 shape, got {shape!r}")
    if int(shape[0]) <= 0 or int(shape[1]) != TOKEN_DIM:
        raise ValueError(f"{key} must be [N,{TOKEN_DIM}], got {shape!r}")


def _as_importance(value: Any, expected_tokens: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("context importance must be a torch.Tensor")
    importance = value.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    if int(importance.numel()) != int(expected_tokens):
        raise ValueError(f"context importance length {importance.numel()} != {expected_tokens}")
    return importance.contiguous()


def _bounded_topk(value: Any, maximum: int) -> int:
    topk = int(value)
    if topk < 0:
        raise ValueError("topk must be non-negative")
    return min(topk, maximum)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
