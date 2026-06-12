"""Generate Step30B trained-predictor occlusion teacher labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data.bridgedata_v2_tfds_importance_proxy import minmax_per_sample
from data.bridgedata_v2_tfds_teacher_label_manifest import (
    METHOD,
    summarize_teacher_importance_manifest,
    write_teacher_importance_manifest_jsonl,
)
from models.bridgedata_v2_trained_predictor_teacher import (
    CONTEXT_FRAMES,
    TOKENS_PER_FRAME,
    CurrentConditionedContextAttentionPredictor,
    future_token_summary,
)
from training.bridgedata_v2_tfds_occlusion_teacher_trainer import (
    resolve_device,
    train_step30b_teachers_from_config,
    write_json,
)


def generate_step30b_occlusion_teacher_labels(
    config_path: str | Path,
    training_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = training_bundle or train_step30b_teachers_from_config(config_path)
    config = bundle["config"]
    train_summary = bundle["teacher_train_summary"]
    if train_summary.get("safe_stop"):
        summary = _safe_stop_summary(config, train_summary.get("reason") or "teacher training safe-stopped")
        write_json(Path(config["output"]["teacher_importance_summary_json"]), summary)
        return {"teacher_importance_summary": summary, "records": []}

    primary_seed = int(config["teacher_training"].get("split_seed", 42))
    teacher = bundle["teachers"][primary_seed]
    device = resolve_device(
        str(config["teacher_training"].get("device", "cpu")),
        bool(config["teacher_training"].get("allow_cpu_fallback", True)),
    )
    teacher = teacher.to(device).eval()
    samples = bundle["samples"][: int(config["occlusion"].get("max_windows", 64))]
    output_dir = Path(config["output"]["teacher_importance_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.pt"):
        old.unlink()
    manifest_path = Path(config["output"]["teacher_importance_manifest_jsonl"])
    if manifest_path.exists():
        manifest_path.unlink()

    records: list[dict[str, Any]] = []
    fallback_count = 0
    raw_means: list[float] = []
    raw_stds: list[float] = []
    norm_mins: list[float] = []
    norm_maxs: list[float] = []
    for sample in samples:
        result = compute_occlusion_importance_for_sample(
            teacher,
            sample,
            device=device,
            occlusion_batch_size=int(config["occlusion"].get("occlusion_batch_size", 512)),
            fallback_to_attention=bool(config["occlusion"].get("fallback_to_attention_weight_label_if_delta_degenerate", True)),
        )
        sample_id = str(sample["sample_id"])
        artifact_path = output_dir / f"{sample_id}_teacher_importance.pt"
        metadata = {
            "token_artifact_path": sample["metadata"]["token_artifact_path"],
            "teacher_split_seed": primary_seed,
            "teacher_training_scope": "small_predictor_teacher_only",
            "current_tokens_kept_full": True,
            "train_current_importance": False,
            "current_importance_generated": False,
            "selector_training_performed": False,
            "action_used_as_input": False,
            "language_used_as_input": False,
            "goal_used_as_input": False,
        }
        torch.save(
            {
                "schema_version": "0.1.0",
                "stage": config["stage"],
                "sample_id": sample_id,
                "trajectory_id": sample.get("trajectory_id"),
                "method": METHOD,
                "context_importance_raw": result["context_importance_raw"],
                "context_importance_norm": result["context_importance_norm"],
                "temporal_importance": result["temporal_importance"],
                "spatial_importance": result["spatial_importance"],
                "stats": result["stats"],
                "metadata": metadata,
            },
            artifact_path,
        )
        stats = result["stats"]
        raw_means.append(float(stats["importance_raw_mean"]))
        raw_stds.append(float(stats["importance_raw_std"]))
        norm_mins.append(float(stats["importance_norm_min"]))
        norm_maxs.append(float(stats["importance_norm_max"]))
        fallback_count += int(bool(stats["fallback_used"]))
        records.append(
            {
                "sample_id": sample_id,
                "trajectory_id": sample.get("trajectory_id"),
                "importance_artifact_path": str(artifact_path),
                "method": METHOD,
                "context_importance_shape": [CONTEXT_FRAMES, TOKENS_PER_FRAME],
                "temporal_importance_shape": [CONTEXT_FRAMES],
                "spatial_importance_shape": [TOKENS_PER_FRAME],
                "current_tokens_kept_full": True,
                "train_current_importance": False,
                "current_importance_generated": False,
                "selector_training_performed": False,
                "action_used_as_input": False,
                "language_used_as_input": False,
                "goal_used_as_input": False,
                "fallback_used": bool(stats["fallback_used"]),
            }
        )

    write_teacher_importance_manifest_jsonl(records, manifest_path)
    manifest_summary = summarize_teacher_importance_manifest(records)
    summary = {
        "stage": config["stage"],
        "safe_stop": False,
        "reason": None,
        "teacher_occlusion_importance_generated": bool(records),
        "num_samples": manifest_summary["num_samples"],
        "num_importance_artifacts": manifest_summary["num_importance_artifacts"],
        "method": METHOD,
        "context_importance_shape_example": manifest_summary["context_importance_shape_example"],
        "temporal_importance_shape_example": manifest_summary["temporal_importance_shape_example"],
        "spatial_importance_shape_example": manifest_summary["spatial_importance_shape_example"],
        "importance_raw_mean": _mean(raw_means),
        "importance_raw_std": _mean(raw_stds),
        "importance_norm_min": min(norm_mins) if norm_mins else None,
        "importance_norm_max": max(norm_maxs) if norm_maxs else None,
        "fallback_used_count": int(fallback_count),
        "teacher_training_scope": "small_predictor_teacher_only",
        "current_tokens_kept_full": True,
        "current_importance_generated": False,
        "train_current_importance": False,
        "selector_training_performed": False,
        "data_importance_shards_written": False,
        "token_extraction_performed": False,
        "proxy_importance_regeneration_performed": False,
        "safety_gate_pass": True,
    }
    write_json(Path(config["output"]["teacher_importance_summary_json"]), summary)
    return {"teacher_importance_summary": summary, "records": records}


def compute_occlusion_importance_for_sample(
    teacher: CurrentConditionedContextAttentionPredictor,
    sample: dict[str, Any],
    device: torch.device,
    occlusion_batch_size: int = 512,
    fallback_to_attention: bool = True,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    with torch.no_grad():
        context = sample["context_tokens"].unsqueeze(0).to(device=device, dtype=torch.float32)
        current = sample["current_tokens"].unsqueeze(0).to(device=device, dtype=torch.float32)
        future = sample["future_tokens"].unsqueeze(0).to(device=device, dtype=torch.float32)
        target = future_token_summary(future)
        summary = teacher.summarize(context, current)
        current_summary = summary["current_summary"]
        context_summary = summary["context_summary"]
        attention = summary["attention_weights"].squeeze(0)
        context_flat = summary["context_flat"].squeeze(0)
        base_pred = teacher.forward_from_summaries(current_summary, context_summary)
        base_loss = F.mse_loss(base_pred, target, reduction="mean")
        contributions = attention.unsqueeze(1) * context_flat
        raw_parts: list[torch.Tensor] = []
        for start in range(0, int(contributions.shape[0]), max(1, occlusion_batch_size)):
            chunk = contributions[start : start + max(1, occlusion_batch_size)]
            occluded_context = context_summary.repeat(chunk.shape[0], 1) - chunk
            pred = teacher.forward_from_summaries(current_summary.repeat(chunk.shape[0], 1), occluded_context)
            losses = ((pred - target.repeat(chunk.shape[0], 1)) ** 2).mean(dim=1)
            raw_parts.append(torch.relu(losses - base_loss))
        raw_flat = torch.cat(raw_parts, dim=0)
        fallback_used = False
        if float(raw_flat.max().item()) <= eps and fallback_to_attention:
            raw_flat = attention.detach().clone()
            fallback_used = True
        raw = raw_flat.reshape(CONTEXT_FRAMES, TOKENS_PER_FRAME).detach().cpu().to(dtype=torch.float32)
        norm = minmax_per_sample(raw, eps=eps)
        temporal = norm.mean(dim=1)
        spatial = norm.mean(dim=0)
        return {
            "method": METHOD,
            "context_importance_raw": raw.contiguous(),
            "context_importance_norm": norm.contiguous(),
            "temporal_importance": temporal.contiguous(),
            "spatial_importance": spatial.contiguous(),
            "stats": {
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
                "current_tokens_kept_full": True,
                "train_current_importance": False,
            },
        }


def _safe_stop_summary(config: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "teacher_occlusion_importance_generated": False,
        "num_samples": 0,
        "context_importance_shape_example": None,
        "fallback_used_count": 0,
        "current_importance_generated": False,
        "train_current_importance": False,
        "selector_training_performed": False,
        "data_importance_shards_written": False,
        "safety_gate_pass": True,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
