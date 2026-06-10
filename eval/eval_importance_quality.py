"""Evaluate whether predictive importance recovers structured toy key tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.importance_shards import load_importance_shard
from data.token_shards import load_token_shard


def _mask_from_token_shard(token_shard: dict[str, Any]) -> torch.Tensor:
    aux_mask = token_shard.get("aux_labels", {}).get("key_token_mask")
    if isinstance(aux_mask, torch.Tensor):
        return aux_mask.float()

    batch_size, num_tokens = token_shard["past_tokens"].shape[:2]
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.float32)
    for row, metadata in enumerate(token_shard["metadata"]):
        for index in metadata.get("key_token_indices", []):
            mask[row, int(index)] = 1.0
    return mask


def _rank_score(scores: torch.Tensor, mask: torch.Tensor) -> float:
    values = []
    for sample_scores, sample_mask in zip(scores, mask):
        key_scores = sample_scores[sample_mask.bool()]
        non_key_scores = sample_scores[~sample_mask.bool()]
        if key_scores.numel() == 0 or non_key_scores.numel() == 0:
            continue
        wins = (key_scores[:, None] > non_key_scores[None, :]).float().mean()
        ties = (key_scores[:, None] == non_key_scores[None, :]).float().mean() * 0.5
        values.append(float((wins + ties).item()))
    return float(sum(values) / len(values)) if values else 0.0


def compute_quality_metrics(scores: torch.Tensor, key_mask: torch.Tensor) -> dict[str, Any]:
    if scores.shape != key_mask.shape:
        raise ValueError(f"scores shape {tuple(scores.shape)} != key_mask shape {tuple(key_mask.shape)}")
    scores = scores.float()
    key_mask = key_mask.float()
    num_samples, num_tokens = scores.shape
    num_key_tokens = int(key_mask[0].sum().item()) if num_samples else 0

    key_values = scores[key_mask.bool()]
    non_key_values = scores[~key_mask.bool()]
    top1_hits = []
    topk_hits = []
    for sample_scores, sample_mask in zip(scores, key_mask):
        key_count = int(sample_mask.sum().item())
        if key_count == 0:
            continue
        top1 = torch.topk(sample_scores, k=1).indices
        topk = torch.topk(sample_scores, k=key_count).indices
        top1_hits.append(float(sample_mask[top1].max().item()))
        topk_hits.append(float(sample_mask[topk].sum().item() / key_count))

    return {
        "num_samples": int(num_samples),
        "num_tokens": int(num_tokens),
        "num_key_tokens": int(num_key_tokens),
        "importance_mean": float(scores.mean().item()),
        "importance_std": float(scores.std(unbiased=False).item()) if scores.numel() > 1 else 0.0,
        "importance_min": float(scores.min().item()) if scores.numel() else 0.0,
        "importance_max": float(scores.max().item()) if scores.numel() else 0.0,
        "positive_importance_ratio": float((scores > 0).float().mean().item()) if scores.numel() else 0.0,
        "key_token_importance_mean": float(key_values.mean().item()) if key_values.numel() else 0.0,
        "non_key_token_importance_mean": float(non_key_values.mean().item()) if non_key_values.numel() else 0.0,
        "key_vs_non_key_gap": (
            float(key_values.mean().item() - non_key_values.mean().item())
            if key_values.numel() and non_key_values.numel()
            else 0.0
        ),
        "top1_hit_rate": float(sum(top1_hits) / len(top1_hits)) if top1_hits else 0.0,
        "topk_hit_rate": float(sum(topk_hits) / len(topk_hits)) if topk_hits else 0.0,
        "auc_like_rank_score": _rank_score(scores, key_mask),
    }


def evaluate_importance_quality(
    token_shard_dir: str | Path,
    importance_shard_dir: str | Path,
    score_field: str = "importance_scores",
) -> dict[str, Any]:
    token_paths = sorted(Path(token_shard_dir).glob("tokens_shard_*.pt"))
    importance_paths = sorted(Path(importance_shard_dir).glob("importance_shard_*.pt"))
    if not token_paths:
        raise FileNotFoundError(f"No token shards found in {token_shard_dir}")
    if len(token_paths) != len(importance_paths):
        raise ValueError(
            f"Token shard count {len(token_paths)} != importance shard count {len(importance_paths)}"
        )

    all_scores = []
    all_norm_scores = []
    all_masks = []
    for token_path, importance_path in zip(token_paths, importance_paths):
        token_shard = load_token_shard(token_path, map_location="cpu")
        importance_shard = load_importance_shard(importance_path, map_location="cpu")
        all_scores.append(importance_shard[score_field].float())
        all_norm_scores.append(importance_shard["importance_scores_norm"].float())
        all_masks.append(_mask_from_token_shard(token_shard))

    scores = torch.cat(all_scores, dim=0)
    norm_scores = torch.cat(all_norm_scores, dim=0)
    masks = torch.cat(all_masks, dim=0)
    metrics = compute_quality_metrics(scores, masks)
    norm_metrics = compute_quality_metrics(norm_scores, masks)
    metrics.update(
        {
            "score_field": score_field,
            "normalized_importance_mean": norm_metrics["importance_mean"],
            "normalized_importance_std": norm_metrics["importance_std"],
            "normalized_importance_min": norm_metrics["importance_min"],
            "normalized_importance_max": norm_metrics["importance_max"],
            "normalized_topk_hit_rate": norm_metrics["topk_hit_rate"],
            "token_shard_dir": str(token_shard_dir),
            "importance_shard_dir": str(importance_shard_dir),
            "importance_shard_files": [path.name for path in importance_paths],
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-shard-dir", required=True)
    parser.add_argument("--importance-shard-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--score-field", default="importance_scores")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_importance_quality(
        args.token_shard_dir,
        args.importance_shard_dir,
        score_field=args.score_field,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("IMPORTANCE_QUALITY_SUMMARY_JSON")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
