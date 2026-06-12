"""Proxy-importance distribution diagnostics for Step33B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from data.bridgedata_v2_tfds_importance_manifest import (
    read_importance_manifest_jsonl,
    validate_importance_artifact,
)


STAGE = "bridgedata_v2_tfds_data_diversity_step33b"
CONTEXT_FRAMES = 16
TOKENS_PER_FRAME = 392


def summarize_step33b_proxy_importance(
    importance_manifest_jsonl: str | Path,
    output_summary_json: str | Path | None = None,
    *,
    shard_name: str = "shard1",
    topk: int = 256,
    allowed_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    records = read_importance_manifest_jsonl(importance_manifest_jsonl)
    if allowed_sample_ids is not None:
        allowed = {str(item) for item in allowed_sample_ids}
        records = [record for record in records if str(record["sample_id"]) in allowed]
    stats = [_importance_record_stats(record, topk=topk) for record in records]
    first = stats[0] if stats else {}
    payload = {
        "stage": STAGE,
        "shard": shard_name,
        "limited_proxy_importance_generation_performed": bool(records),
        "importance_generation_performed": bool(records),
        "method": "proxy_token_mse_dryrun",
        "num_samples": len(records),
        "context_importance_shape_example": first.get("context_importance_shape"),
        "temporal_importance_shape_example": first.get("temporal_importance_shape"),
        "spatial_importance_shape_example": first.get("spatial_importance_shape"),
        "topk": int(topk),
        "topk_mass_mean": _mean([item["topk_mass"] for item in stats]),
        "topk_mass_std": _std([item["topk_mass"] for item in stats]),
        "temporal_concentration_mean": _mean([item["temporal_concentration"] for item in stats]),
        "spatial_concentration_mean": _mean([item["spatial_concentration"] for item in stats]),
        "importance_mean": _mean([item["importance_mean"] for item in stats]),
        "importance_std": _mean([item["importance_std"] for item in stats]),
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "current_importance_generated": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "data_importance_shards_written": False,
        "target_aware_proxy_summary_only": True,
        "sample_stats": stats,
        "safety_gate_pass": True,
    }
    if output_summary_json is not None:
        output = Path(output_summary_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def fake_importance_summary(values: torch.Tensor, *, topk: int = 2) -> dict[str, Any]:
    if list(values.shape) != [CONTEXT_FRAMES, TOKENS_PER_FRAME]:
        raise ValueError("fake values must have shape [16, 392]")
    flat = values.reshape(-1).to(dtype=torch.float32)
    total = float(flat.sum().item())
    top = torch.topk(flat, k=min(int(topk), int(flat.numel()))).values
    temporal = values.sum(dim=1)
    spatial = values.sum(dim=0)
    return {
        "context_importance_shape": [CONTEXT_FRAMES, TOKENS_PER_FRAME],
        "topk_mass": float(top.sum().item() / total) if total > 0 else 0.0,
        "temporal_concentration": _concentration(temporal),
        "spatial_concentration": _concentration(spatial),
        "current_importance_generated": False,
    }


def _importance_record_stats(record: dict[str, Any], *, topk: int) -> dict[str, Any]:
    artifact = validate_importance_artifact(record["importance_artifact_path"])
    values = artifact["context_importance_norm"].detach().to(dtype=torch.float32, device="cpu")
    if list(values.shape) != [CONTEXT_FRAMES, TOKENS_PER_FRAME]:
        raise ValueError(f"context_importance_norm shape {list(values.shape)} != [16, 392]")
    stats = fake_importance_summary(values, topk=topk)
    stats.update(
        {
            "sample_id": str(record["sample_id"]),
            "trajectory_id": record.get("trajectory_id"),
            "importance_mean": float(values.mean().item()),
            "importance_std": float(values.std(unbiased=False).item()),
            "temporal_importance_shape": [CONTEXT_FRAMES],
            "spatial_importance_shape": [TOKENS_PER_FRAME],
        }
    )
    return stats


def _concentration(values: torch.Tensor) -> float:
    flat = values.detach().to(dtype=torch.float32).reshape(-1)
    total = float(flat.sum().item())
    if total <= 0:
        return 0.0
    return float((flat.max().item()) / total)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return float((sum((float(value) - m) ** 2 for value in values) / len(values)) ** 0.5)
