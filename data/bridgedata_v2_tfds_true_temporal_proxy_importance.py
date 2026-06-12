"""Proxy importance for Step33A shape-flexible true-temporal tokens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data.bridgedata_v2_tfds_true_temporal_schema import TOKEN_DIM


METHOD = "proxy_token_mse_dryrun_true_temporal"
SPATIAL_TOKENS_PER_BIN = 196


def compute_true_temporal_context_importance(
    context_tokens: torch.Tensor,
    current_summary: torch.Tensor,
    future_summary: torch.Tensor,
    *,
    context_weight: float = 0.25,
    current_weight: float = 0.75,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    """Analytic no-training leave-one-token proxy for flat clip tokens."""

    with torch.no_grad():
        context = _as_context_tokens(context_tokens)
        current = _as_summary(current_summary, "current_summary")
        future = _as_summary(future_summary, "future_summary")
        count = int(context.shape[0])
        if count <= 1:
            raise ValueError("context token count must be greater than one")
        context_sum = context.sum(dim=0)
        context_mean = context_sum / count
        base_pred = current_weight * current + context_weight * context_mean
        base_loss = F.mse_loss(base_pred, future, reduction="mean")

        masked_mean = (context_sum.unsqueeze(0) - context) / (count - 1)
        masked_pred = current_weight * current.unsqueeze(0) + context_weight * masked_mean
        masked_losses = ((masked_pred - future.unsqueeze(0)) ** 2).mean(dim=1)
        raw = torch.relu(masked_losses - base_loss)
        fallback_used = False
        if float(raw.max().item()) <= eps:
            raw = _cosine_relevance_fallback(context, future)
            fallback_used = True
        norm = minmax_per_sample(raw, eps=eps)
        bin_info = temporal_spatial_summary(norm)
        stats = {
            "method": METHOD,
            "base_loss": float(base_loss.item()),
            "importance_raw_mean": float(raw.mean().item()),
            "importance_raw_std": float(raw.std(unbiased=False).item()),
            "importance_raw_min": float(raw.min().item()),
            "importance_raw_max": float(raw.max().item()),
            "importance_norm_min": float(norm.min().item()),
            "importance_norm_max": float(norm.max().item()),
            "importance_norm_mean": float(norm.mean().item()),
            "importance_norm_std": float(norm.std(unbiased=False).item()),
            "fallback_used": bool(fallback_used),
            "num_context_tokens": count,
            "current_tokens_kept_full": True,
            "train_current_importance": False,
            "no_trainable_parameters": True,
            **bin_info["metadata"],
        }
        return {
            "method": METHOD,
            "context_importance_raw": raw.detach().cpu().contiguous(),
            "context_importance_norm": norm.detach().cpu().contiguous(),
            "temporal_importance": bin_info["temporal_importance"],
            "spatial_importance": bin_info["spatial_importance"],
            "temporal_spatial_summary_available": bin_info["temporal_spatial_summary_available"],
            "stats": stats,
        }


def temporal_spatial_summary(importance: torch.Tensor) -> dict[str, Any]:
    flat = importance.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    count = int(flat.numel())
    if count > 0 and count % SPATIAL_TOKENS_PER_BIN == 0:
        bins = count // SPATIAL_TOKENS_PER_BIN
        matrix = flat.reshape(bins, SPATIAL_TOKENS_PER_BIN)
        temporal = matrix.mean(dim=1).contiguous()
        spatial = matrix.mean(dim=0).contiguous()
        return {
            "temporal_spatial_summary_available": True,
            "temporal_importance": temporal,
            "spatial_importance": spatial,
            "metadata": {
                "temporal_bins_available": True,
                "num_temporal_bins": bins,
                "num_spatial_tokens_per_bin": SPATIAL_TOKENS_PER_BIN,
            },
        }
    return {
        "temporal_spatial_summary_available": False,
        "temporal_importance": None,
        "spatial_importance": None,
        "metadata": {
            "temporal_bins_available": False,
            "reason": "model output shape is treated as flat clip tokens",
        },
    }


def write_true_temporal_importance_manifest(records: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_true_temporal_importance_record(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return output


def read_true_temporal_importance_manifest(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid Step33A importance manifest line {line_number}: {exc}") from exc
            validate_true_temporal_importance_record(record)
            records.append(record)
    return records


def validate_true_temporal_importance_record(record: dict[str, Any]) -> bool:
    if record.get("method") != METHOD:
        raise ValueError(f"unexpected Step33A importance method {record.get('method')!r}")
    shape = record.get("context_importance_shape")
    if not isinstance(shape, list) or len(shape) != 1 or int(shape[0]) <= 0:
        raise ValueError(f"context_importance_shape must be [N], got {shape!r}")
    for flag, expected in {
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "current_importance_generated": False,
        "selector_training_performed": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
    }.items():
        if bool(record.get(flag, False)) != expected:
            raise ValueError(f"{flag} must be {expected}")
    return True


def load_true_temporal_importance_artifact(record: dict[str, Any]) -> dict[str, Any]:
    validate_true_temporal_importance_record(record)
    artifact_path = Path(str(record["importance_artifact_path"]))
    artifact = _torch_load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError(f"Step33A importance artifact must be a dict: {artifact_path}")
    if artifact.get("method") != METHOD:
        raise ValueError(f"unexpected Step33A importance method {artifact.get('method')!r}")
    context_importance = artifact.get("context_importance_norm")
    if not isinstance(context_importance, torch.Tensor):
        raise ValueError("context_importance_norm missing or not tensor")
    return artifact


def summarize_true_temporal_importance(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_samples": 0,
            "context_importance_shape_example": None,
            "temporal_spatial_summary_available": False,
        }
    for record in records:
        validate_true_temporal_importance_record(record)
    first = records[0]
    return {
        "num_samples": len(records),
        "num_importance_artifacts": len({str(record["importance_artifact_path"]) for record in records}),
        "context_importance_shape_example": first["context_importance_shape"],
        "temporal_importance_shape_example": first.get("temporal_importance_shape"),
        "spatial_importance_shape_example": first.get("spatial_importance_shape"),
        "temporal_spatial_summary_available": bool(first.get("temporal_spatial_summary_available")),
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "current_importance_generated": False,
    }


def minmax_per_sample(values: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    values = values.detach().to(dtype=torch.float32, device="cpu")
    vmin = values.min()
    vmax = values.max()
    denom = vmax - vmin
    if float(denom.abs().item()) <= eps:
        return torch.zeros_like(values)
    return ((values - vmin) / denom).clamp(0.0, 1.0).contiguous()


def _cosine_relevance_fallback(context: torch.Tensor, future_summary: torch.Tensor) -> torch.Tensor:
    context_norm = F.normalize(context.to(dtype=torch.float32), dim=1, eps=1.0e-12)
    future_norm = F.normalize(future_summary.to(dtype=torch.float32).unsqueeze(0), dim=1, eps=1.0e-12)
    return ((context_norm * future_norm).sum(dim=1) + 1.0).mul(0.5).clamp_min(0.0).contiguous()


def _as_context_tokens(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("context_tokens must be a torch.Tensor")
    tokens = value.detach().to(dtype=torch.float32, device="cpu")
    if tokens.ndim != 2 or int(tokens.shape[1]) != TOKEN_DIM or int(tokens.shape[0]) <= 1:
        raise ValueError(f"context_tokens must be [N,{TOKEN_DIM}], got {list(tokens.shape)}")
    return tokens.contiguous()


def _as_summary(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    summary = value.detach().to(dtype=torch.float32, device="cpu")
    if list(summary.shape) != [TOKEN_DIM]:
        raise ValueError(f"{name} must be [{TOKEN_DIM}], got {list(summary.shape)}")
    return summary.contiguous()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
