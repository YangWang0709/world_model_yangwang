"""Evaluate BAIR VideoMAE predictive importance shards."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.importance_shards import load_importance_shard


DEFAULT_IMPORTANCE_ROOT = (
    PROJECT_ROOT / "data" / "importance_shards" / "bair_videomae_teacher_smoke"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance-root", default=str(DEFAULT_IMPORTANCE_ROOT))
    parser.add_argument("--output-path")
    return parser.parse_args()


def _stats(tensor: torch.Tensor, prefix: str) -> dict[str, float]:
    values = tensor.detach().float().reshape(-1)
    if values.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
    }


def _evaluate_shard_paths(
    shard_paths: list[Path],
    importance_dir: str | Path,
    split: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if not shard_paths:
        raise FileNotFoundError(f"No importance shards found for split {split!r} in {importance_dir}")

    start_time = time.perf_counter()
    shards = [load_importance_shard(path, map_location="cpu") for path in shard_paths]
    importance = torch.cat([shard["importance_scores"].float() for shard in shards], dim=0)
    normalized = torch.cat([shard["importance_scores_norm"].float() for shard in shards], dim=0)
    base_losses = torch.cat([shard["base_losses"].float() for shard in shards], dim=0)
    masked_losses = torch.cat([shard["masked_losses"].float() for shard in shards], dim=0)
    top1 = importance.topk(k=1, dim=1).values.mean(dim=1)
    top5 = importance.topk(k=min(5, importance.shape[1]), dim=1).values.mean(dim=1)
    top10 = importance.topk(k=min(10, importance.shape[1]), dim=1).values.mean(dim=1)
    first_teacher_config = dict(shards[0].get("teacher_config", {}))
    token_dim = first_teacher_config.get("token_dim")
    elapsed_time_sec = round(time.perf_counter() - start_time, 3)

    summary = {
        "importance_dir": str(importance_dir),
        "split": split,
        "num_shards": len(shard_paths),
        "num_samples": int(importance.shape[0]),
        "num_tokens": int(importance.shape[1]),
        "token_dim": int(token_dim) if token_dim is not None else None,
        "shard_files": [path.name for path in shard_paths],
        **_stats(importance, "importance"),
        **_stats(normalized, "normalized_importance"),
        "positive_importance_ratio": float((importance > 0).float().mean().item()),
        "base_loss_mean": float(base_losses.mean().item()),
        "base_loss_std": float(base_losses.std(unbiased=False).item()) if base_losses.numel() > 1 else 0.0,
        "base_loss_min": float(base_losses.min().item()),
        "base_loss_max": float(base_losses.max().item()),
        "masked_loss_mean": float(masked_losses.mean().item()),
        "masked_loss_std": float(masked_losses.std(unbiased=False).item()) if masked_losses.numel() > 1 else 0.0,
        "masked_loss_min": float(masked_losses.min().item()),
        "masked_loss_max": float(masked_losses.max().item()),
        "top1_importance_mean": float(top1.mean().item()),
        "top5_importance_mean": float(top5.mean().item()),
        "top10_importance_mean": float(top10.mean().item()),
        "elapsed_time_sec": elapsed_time_sec,
    }

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_bair_importance_root(
    importance_root: str | Path = DEFAULT_IMPORTANCE_ROOT,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(importance_root)
    split_summaries: dict[str, dict[str, Any]] = {}
    all_paths: list[Path] = []
    for split in ("train", "test"):
        split_dir = root / split
        shard_paths = sorted(split_dir.glob("importance_shard_*.pt"))
        split_summaries[split] = _evaluate_shard_paths(
            shard_paths,
            importance_dir=split_dir,
            split=split,
            output_path=split_dir / "eval_importance_summary.json",
        )
        all_paths.extend(shard_paths)

    overall_output = Path(output_path) if output_path else root / "eval_importance_summary.json"
    overall = _evaluate_shard_paths(
        all_paths,
        importance_dir=root,
        split="overall",
        output_path=overall_output,
    )
    overall["split_summaries"] = split_summaries
    overall_output.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps(overall, indent=2))
    print(f"EVAL_IMPORTANCE_SUMMARY_WRITTEN = {overall_output}")
    return overall


def main() -> None:
    args = parse_args()
    evaluate_bair_importance_root(args.importance_root, output_path=args.output_path)


if __name__ == "__main__":
    main()
