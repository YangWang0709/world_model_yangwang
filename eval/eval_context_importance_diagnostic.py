"""Diagnose Step17 context-importance labels for Step18 oracle-gap analysis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.importance_shards import load_importance_shard


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "context_selector_oracle_gap_bair_1000_128.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return payload


def _resolve_shards(root: str | Path, shard_glob: str = "importance_shard_*.pt") -> list[Path]:
    path = Path(root)
    paths = sorted(path.glob(shard_glob)) if path.is_dir() else [path]
    if not paths:
        raise FileNotFoundError(f"No context importance shards found under {root}")
    missing = [str(item) for item in paths if not item.exists()]
    if missing:
        raise FileNotFoundError(f"Missing context importance shards: {missing}")
    return paths


def _take_max_samples(tensor: torch.Tensor, max_samples: int | None) -> torch.Tensor:
    if max_samples is None:
        return tensor
    return tensor[: int(max_samples)]


def load_importance_tensors(
    importance_dir: str | Path,
    *,
    max_samples: int | None = None,
    shard_glob: str = "importance_shard_*.pt",
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_parts: list[torch.Tensor] = []
    norm_parts: list[torch.Tensor] = []
    loaded = 0
    for shard_path in _resolve_shards(importance_dir, shard_glob=shard_glob):
        shard = load_importance_shard(shard_path, map_location="cpu")
        raw = shard["importance_scores"].float().cpu()
        norm = shard["importance_scores_norm"].float().cpu()
        if max_samples is not None:
            remaining = int(max_samples) - loaded
            if remaining <= 0:
                break
            raw = raw[:remaining]
            norm = norm[:remaining]
        raw_parts.append(raw)
        norm_parts.append(norm)
        loaded += int(raw.shape[0])
    if not raw_parts:
        raise ValueError("No context importance samples loaded")
    return torch.cat(raw_parts, dim=0), torch.cat(norm_parts, dim=0)


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def temporal_block_ids(num_tokens: int, temporal_blocks: int = 8) -> torch.Tensor:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if temporal_blocks <= 0:
        raise ValueError("temporal_blocks must be positive")
    return torch.arange(num_tokens).mul(int(temporal_blocks)).floor_divide(int(num_tokens)).clamp_max(temporal_blocks - 1)


def _random_selected_mean(values: torch.Tensor, topk: int, trials: int, seed: int) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    rows = []
    for _ in range(int(trials)):
        selected = []
        for row in values:
            indices = torch.randperm(row.numel(), generator=generator)[:topk]
            selected.append(row[indices].mean())
        rows.append(torch.stack(selected).mean())
    return float(torch.stack(rows).mean().item()) if rows else 0.0


def _uniform_selected_mean(values: torch.Tensor, topk: int) -> float:
    indices = torch.linspace(0, values.shape[1] - 1, steps=topk).round().long()
    return float(values[:, indices].mean().item())


def compute_topk_concentration(values: torch.Tensor, *, topk_values: list[int], random_trials: int, seed: int) -> dict[str, Any]:
    values = values.float()
    total_positive_mass = values.clamp_min(0.0).sum(dim=1).clamp_min(1e-12)
    out: dict[str, Any] = {}
    for k in topk_values:
        k = min(int(k), int(values.shape[1]))
        top_values = torch.topk(values, k=k, dim=1).values
        top_mean = float(top_values.mean().item())
        mass_ratio = float((top_values.clamp_min(0.0).sum(dim=1) / total_positive_mass).mean().item())
        random_mean = _random_selected_mean(values, k, random_trials, seed)
        out[f"top{k}_importance_mean"] = top_mean
        out[f"top{k}_mass_ratio"] = mass_ratio
        out[f"top{k}_vs_random_mean_gap"] = top_mean - random_mean
    return out


def compute_temporal_block_stats(values: torch.Tensor, *, topk: int, temporal_blocks: int) -> dict[str, Any]:
    block_ids = temporal_block_ids(int(values.shape[1]), temporal_blocks=temporal_blocks)
    top_indices = torch.topk(values, k=topk, dim=1).indices
    block_rows = []
    oracle_counts = []
    for block_id in range(int(temporal_blocks)):
        mask = block_ids == block_id
        block_values = values[:, mask]
        topk_count = (block_ids[top_indices] == block_id).sum(dim=1).float()
        oracle_counts.append(topk_count.mean())
        block_rows.append(
            {
                "block": int(block_id),
                "token_count": int(mask.sum().item()),
                "mean_importance": float(block_values.mean().item()),
                "max_importance": float(block_values.max().item()),
                "positive_ratio": float((block_values > 0).float().mean().item()),
                "oracle_topk_count_mean": float(topk_count.mean().item()),
            }
        )
    expected = float(topk) / float(temporal_blocks)
    coverage = float(sum(1 for item in oracle_counts if float(item.item()) > 0.0) / float(temporal_blocks))
    return {
        "temporal_block_method": "approximate_equal_token_index_blocks",
        "temporal_blocks": int(temporal_blocks),
        "block_stats": block_rows,
        "oracle_topk_block_distribution": [float(item.item()) for item in oracle_counts],
        "random_expected_block_distribution": [expected for _ in range(int(temporal_blocks))],
        "oracle_topk_block_coverage": coverage,
    }


def label_quality_warnings(
    *,
    positive_ratio: float,
    near_zero_ratio: float,
    top32_mass_ratio: float,
    top32_vs_random_gap: float,
    temporal_block_coverage: float,
) -> list[str]:
    warnings: list[str] = []
    if positive_ratio < 0.05:
        warnings.append("context importance labels are sparse: positive ratio is very low")
    if positive_ratio > 0.95:
        warnings.append("context importance labels are diffuse: almost all tokens are positive")
    if near_zero_ratio > 0.50:
        warnings.append("many context importance labels are near zero")
    if top32_mass_ratio < 0.20:
        warnings.append("top32 mass is weakly concentrated; random context may remain competitive")
    if top32_vs_random_gap < 0.05:
        warnings.append("oracle top32 selected importance is close to random")
    if temporal_block_coverage < 0.75:
        warnings.append("oracle topK is temporally uneven; block-balanced selection is worth testing")
    return warnings


def build_context_importance_diagnostic(
    importance_scores: torch.Tensor,
    importance_scores_norm: torch.Tensor,
    *,
    topk_values: list[int] | None = None,
    context_topk: int = 32,
    temporal_blocks: int = 8,
    random_trials: int = 20,
    seed: int = 42,
    step17_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if importance_scores.shape != importance_scores_norm.shape:
        raise ValueError("raw and normalized importance tensors must have the same shape")
    if importance_scores.ndim != 2:
        raise ValueError(f"importance tensors must be [B, N_ctx], got {tuple(importance_scores.shape)}")
    topk_values = topk_values or [8, 16, 32, 64]
    raw = importance_scores.float()
    norm = importance_scores_norm.float().clamp(0.0, 1.0)
    raw_stats = _tensor_stats(raw)
    norm_stats = _tensor_stats(norm)
    positive_ratio = float((raw > 0).float().mean().item())
    negative_ratio = float((raw < 0).float().mean().item())
    near_zero_ratio = float((raw.abs() <= 1e-6).float().mean().item())
    topk = min(int(context_topk), int(norm.shape[1]))
    concentration = compute_topk_concentration(norm, topk_values=topk_values, random_trials=random_trials, seed=seed)
    temporal = compute_temporal_block_stats(norm, topk=topk, temporal_blocks=temporal_blocks)
    oracle_mean = float(torch.topk(norm, k=topk, dim=1).values.mean().item())
    random_mean = _random_selected_mean(norm, topk, random_trials, seed)
    uniform_mean = _uniform_selected_mean(norm, topk)
    reference = step17_reference or {}
    teacher_mse = reference.get("teacher_context_importance_topk_mse")
    random_mse = reference.get("random_context_topk_mse")
    oracle_random_downstream_gap = None
    if teacher_mse is not None and random_mse is not None and math.isfinite(float(teacher_mse)) and math.isfinite(float(random_mse)):
        oracle_random_downstream_gap = float(random_mse) - float(teacher_mse)
    warnings = label_quality_warnings(
        positive_ratio=positive_ratio,
        near_zero_ratio=near_zero_ratio,
        top32_mass_ratio=float(concentration.get("top32_mass_ratio", 0.0)),
        top32_vs_random_gap=float(concentration.get("top32_vs_random_mean_gap", 0.0)),
        temporal_block_coverage=float(temporal.get("oracle_topk_block_coverage", 0.0)),
    )
    return {
        "num_samples": int(norm.shape[0]),
        "num_context_tokens": int(norm.shape[1]),
        "importance_stats": raw_stats,
        "importance_norm_stats": norm_stats,
        "positive_importance_ratio": positive_ratio,
        "negative_importance_ratio": negative_ratio,
        "near_zero_ratio": near_zero_ratio,
        "topk_concentration": concentration,
        "temporal_block_stats": temporal,
        "oracle_random_gap": {
            "teacher_context_importance_topk_selected_importance_mean": oracle_mean,
            "random_context_topk_selected_importance_mean": random_mean,
            "uniform_context_topk_selected_importance_mean": uniform_mean,
            "oracle_vs_random_importance_gap": oracle_mean - random_mean,
            "oracle_vs_uniform_importance_gap": oracle_mean - uniform_mean,
            "oracle_vs_random_downstream_mse_gap": oracle_random_downstream_gap,
        },
        "label_quality_warnings": warnings,
        "label_sparse_or_noisy": bool(warnings),
    }


def render_context_importance_diagnostic_markdown(summary: dict[str, Any]) -> str:
    stats = summary["importance_stats"]
    norm = summary["importance_norm_stats"]
    gap = summary["oracle_random_gap"]
    temporal = summary["temporal_block_stats"]
    lines = [
        "# Step18 Context Importance Diagnostic",
        "",
        f"- samples: `{summary['num_samples']}`",
        f"- context tokens: `{summary['num_context_tokens']}`",
        f"- raw importance mean/std/min/max: `{stats['mean']:.8f}` / `{stats['std']:.8f}` / `{stats['min']:.8f}` / `{stats['max']:.8f}`",
        f"- norm importance mean/std/min/max: `{norm['mean']:.8f}` / `{norm['std']:.8f}` / `{norm['min']:.8f}` / `{norm['max']:.8f}`",
        f"- positive ratio: `{summary['positive_importance_ratio']:.6f}`",
        f"- negative ratio: `{summary['negative_importance_ratio']:.6f}`",
        f"- near-zero ratio: `{summary['near_zero_ratio']:.6f}`",
        "",
        "## TopK Concentration",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in sorted(summary["topk_concentration"].items()):
        lines.append(f"| {key} | {float(value):.8f} |")
    lines.extend(
        [
            "",
            "## Temporal Blocks",
            "",
            f"- method: `{temporal['temporal_block_method']}`",
            f"- oracle topK block distribution: `{[round(float(v), 4) for v in temporal['oracle_topk_block_distribution']]}`",
            f"- random expected block distribution: `{[round(float(v), 4) for v in temporal['random_expected_block_distribution']]}`",
            "",
            "| block | token_count | mean | max | positive_ratio | oracle_topK_count |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in temporal["block_stats"]:
        lines.append(
            f"| {row['block']} | {row['token_count']} | {row['mean_importance']:.8f} | "
            f"{row['max_importance']:.8f} | {row['positive_ratio']:.6f} | {row['oracle_topk_count_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Oracle vs Random",
            "",
            f"- oracle selected importance mean: `{gap['teacher_context_importance_topk_selected_importance_mean']:.8f}`",
            f"- random selected importance mean: `{gap['random_context_topk_selected_importance_mean']:.8f}`",
            f"- uniform selected importance mean: `{gap['uniform_context_topk_selected_importance_mean']:.8f}`",
            f"- oracle vs random importance gap: `{gap['oracle_vs_random_importance_gap']:.8f}`",
            f"- oracle vs uniform importance gap: `{gap['oracle_vs_uniform_importance_gap']:.8f}`",
            f"- oracle vs random downstream MSE gap: `{gap['oracle_vs_random_downstream_mse_gap']}`",
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in summary.get("label_quality_warnings", [])] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def write_context_importance_diagnostic(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    diag_cfg = config.get("label_diagnostic", {})
    output_cfg = config["output"]
    run_dir = Path(output_cfg["run_root"]) / output_cfg["run_name"]
    out_dir = run_dir / "label_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, norm = load_importance_tensors(
        data_cfg["context_importance_shard_dir_train"],
        max_samples=int(data_cfg.get("max_train_samples", 0)) or None,
        shard_glob=data_cfg.get("importance_shard_glob", "importance_shard_*.pt"),
    )
    summary = build_context_importance_diagnostic(
        raw,
        norm,
        topk_values=[int(k) for k in diag_cfg.get("topk_values", [8, 16, 32, 64])],
        context_topk=int(config["selection"]["context_topk"]),
        temporal_blocks=int(diag_cfg.get("temporal_blocks", 8)),
        random_trials=int(diag_cfg.get("random_trials", 20)),
        seed=int(config.get("seed", 42)),
        step17_reference=config.get("step17_reference", {}),
    )
    (out_dir / "context_importance_diagnostic.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "context_importance_diagnostic.md").write_text(render_context_importance_diagnostic_markdown(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    summary = write_context_importance_diagnostic(load_yaml(parse_args().config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
