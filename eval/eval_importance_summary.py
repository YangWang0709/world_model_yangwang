"""Aggregate statistics over predictive importance shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.importance_shards import load_importance_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance-dir", required=True)
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


def evaluate_importance_dir(importance_dir: str | Path, output_path: str | Path | None = None) -> dict:
    root = Path(importance_dir)
    shard_paths = sorted(root.glob("importance_shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No importance shards found in {root}")

    shards = [load_importance_shard(path, map_location="cpu") for path in shard_paths]
    importance = torch.cat([shard["importance_scores"].float() for shard in shards], dim=0)
    normalized = torch.cat([shard["importance_scores_norm"].float() for shard in shards], dim=0)
    base_losses = torch.cat([shard["base_losses"].float() for shard in shards], dim=0)
    masked_losses = torch.cat([shard["masked_losses"].float() for shard in shards], dim=0)
    top1 = importance.topk(k=1, dim=1).values.mean(dim=1)
    top5 = importance.topk(k=min(5, importance.shape[1]), dim=1).values.mean(dim=1)

    summary = {
        "importance_dir": str(root),
        "num_shards": len(shard_paths),
        "num_samples": int(importance.shape[0]),
        "num_tokens": int(importance.shape[1]),
        "shard_files": [path.name for path in shard_paths],
        **_stats(importance, "importance"),
        **_stats(normalized, "normalized_importance"),
        "base_loss_mean": float(base_losses.mean().item()),
        "masked_loss_mean": float(masked_losses.mean().item()),
        "top1_importance_mean": float(top1.mean().item()),
        "top5_importance_mean": float(top5.mean().item()),
    }

    destination = Path(output_path) if output_path else root / "eval_importance_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"EVAL_IMPORTANCE_SUMMARY_WRITTEN = {destination}")
    return summary


def main() -> None:
    args = parse_args()
    evaluate_importance_dir(args.importance_dir, args.output_path)


if __name__ == "__main__":
    main()
