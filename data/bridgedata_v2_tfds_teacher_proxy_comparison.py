"""Compare Step30B trained-teacher labels against Step30A proxy labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from data.bridgedata_v2_tfds_importance_manifest import read_importance_manifest_jsonl, validate_importance_artifact
from data.bridgedata_v2_tfds_teacher_label_manifest import (
    read_teacher_importance_manifest_jsonl,
    validate_teacher_importance_artifact,
)


def compare_teacher_proxy_importance(
    teacher_manifest_jsonl: str | Path,
    proxy_manifest_jsonl: str | Path,
    output_json: str | Path | None = None,
    topk_values: list[int] | tuple[int, ...] = (64, 128, 256, 512),
) -> dict[str, Any]:
    teacher_records = read_teacher_importance_manifest_jsonl(teacher_manifest_jsonl)
    proxy_records = read_importance_manifest_jsonl(proxy_manifest_jsonl)
    proxy_by_id = {str(record["sample_id"]): record for record in proxy_records}
    pearsons: list[float] = []
    spearmans: list[float] = []
    teacher_stds: list[float] = []
    topk_overlap: dict[int, list[float]] = {int(k): [] for k in topk_values}
    temporal_corrs: list[float] = []
    spatial_corrs: list[float] = []
    teacher_mass_concentration: list[float] = []
    proxy_mass_under_teacher_topk: list[float] = []
    teacher_mass_under_proxy_topk: list[float] = []

    for record in teacher_records:
        sample_id = str(record["sample_id"])
        if sample_id not in proxy_by_id:
            continue
        teacher = validate_teacher_importance_artifact(record["importance_artifact_path"])
        proxy = validate_importance_artifact(proxy_by_id[sample_id]["importance_artifact_path"])
        teacher_flat = teacher["context_importance_norm"].reshape(-1).to(dtype=torch.float32)
        proxy_flat = proxy["context_importance_norm"].reshape(-1).to(dtype=torch.float32)
        pearsons.append(pearson_corr(teacher_flat, proxy_flat))
        spearmans.append(spearman_corr(teacher_flat, proxy_flat))
        teacher_stds.append(float(teacher_flat.std(unbiased=False).item()))
        temporal_corrs.append(pearson_corr(teacher["temporal_importance"], proxy["temporal_importance"]))
        spatial_corrs.append(pearson_corr(teacher["spatial_importance"], proxy["spatial_importance"]))
        for topk in topk_values:
            k = min(int(topk), int(teacher_flat.numel()))
            teacher_idx = torch.topk(teacher_flat, k=k, largest=True).indices
            proxy_idx = torch.topk(proxy_flat, k=k, largest=True).indices
            topk_overlap[int(topk)].append(float(len(set(teacher_idx.tolist()) & set(proxy_idx.tolist())) / k))
        k256 = min(256, int(teacher_flat.numel()))
        teacher_idx = torch.topk(teacher_flat, k=k256, largest=True).indices
        proxy_idx = torch.topk(proxy_flat, k=k256, largest=True).indices
        teacher_total = float(teacher_flat.sum().item())
        proxy_total = float(proxy_flat.sum().item())
        teacher_mass_concentration.append(float(teacher_flat[teacher_idx].sum().item() / teacher_total) if teacher_total > 0 else 0.0)
        proxy_mass_under_teacher_topk.append(float(proxy_flat[teacher_idx].sum().item() / proxy_total) if proxy_total > 0 else 0.0)
        teacher_mass_under_proxy_topk.append(float(teacher_flat[proxy_idx].sum().item() / teacher_total) if teacher_total > 0 else 0.0)

    comparison = {
        "stage": "bridgedata_v2_tfds_occlusion_teacher_step30b",
        "teacher_proxy_comparison_performed": bool(pearsons),
        "num_samples": len(pearsons),
        "teacher_proxy_pearson_mean": mean(pearsons),
        "teacher_proxy_spearman_mean": mean(spearmans),
        "topk_overlap": {str(k): mean(values) for k, values in topk_overlap.items()},
        "teacher_mass_concentration_top256_mean": mean(teacher_mass_concentration),
        "proxy_mass_under_teacher_top256_mean": mean(proxy_mass_under_teacher_topk),
        "teacher_mass_under_proxy_top256_mean": mean(teacher_mass_under_proxy_topk),
        "temporal_correlation_mean": mean(temporal_corrs),
        "spatial_correlation_mean": mean(spatial_corrs),
        "teacher_label_nontrivial": bool(teacher_stds and mean(teacher_stds) > 1.0e-8),
        "teacher_label_stability_across_seeds": "not_computed",
        "interpretation": "Low teacher/proxy correlation is not a failure; it can indicate a distinct trained-teacher label.",
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "safety_gate_pass": True,
    }
    if output_json is not None:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    return comparison


def pearson_corr(left: torch.Tensor, right: torch.Tensor, eps: float = 1.0e-12) -> float:
    x = left.detach().to(dtype=torch.float32).reshape(-1)
    y = right.detach().to(dtype=torch.float32).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if float(denom.item()) <= eps:
        return 0.0
    return float(torch.dot(x, y).div(denom).clamp(-1.0, 1.0).item())


def spearman_corr(left: torch.Tensor, right: torch.Tensor) -> float:
    if float(left.detach().to(dtype=torch.float32).std(unbiased=False).item()) <= 1.0e-12:
        return 0.0
    if float(right.detach().to(dtype=torch.float32).std(unbiased=False).item()) <= 1.0e-12:
        return 0.0
    return pearson_corr(rank_tensor(left), rank_tensor(right))


def rank_tensor(value: torch.Tensor) -> torch.Tensor:
    flat = value.detach().to(dtype=torch.float32).reshape(-1)
    order = torch.argsort(flat, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(flat.numel(), dtype=torch.float32)
    return ranks


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))
