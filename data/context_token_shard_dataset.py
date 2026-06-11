"""Context/current/future token shards for Step 17 context bottlenecks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from data.importance_shards import load_importance_shard


CONTEXT_TOKEN_SHARD_SCHEMA_VERSION = "0.1.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_paths(paths: str | Path | Iterable[str | Path], shard_glob: str) -> list[Path]:
    if isinstance(paths, (str, Path)):
        root = Path(paths)
        resolved = sorted(root.glob(shard_glob)) if root.is_dir() else [root]
    else:
        resolved = [Path(path) for path in paths]
    if not resolved:
        raise FileNotFoundError(f"No context token shards found for {paths!r} with glob {shard_glob!r}")
    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Context token shards do not exist: {missing}")
    return resolved


def validate_context_token_shard(shard: dict[str, Any], strict: bool = True) -> bool:
    required = (
        "schema_version",
        "encoder_name",
        "encoder_config",
        "created_at",
        "split",
        "sample_ids",
        "task_texts",
        "context_tokens",
        "current_tokens",
        "future_tokens",
        "metadata",
    )
    missing = [key for key in required if key not in shard]
    if missing:
        raise KeyError(f"context token shard missing required keys: {missing}")
    if shard["schema_version"] != CONTEXT_TOKEN_SHARD_SCHEMA_VERSION:
        raise ValueError(f"unsupported context token schema {shard['schema_version']!r}")
    context = shard["context_tokens"]
    current = shard["current_tokens"]
    future = shard["future_tokens"]
    for key, value in (("context_tokens", context), ("current_tokens", current), ("future_tokens", future)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor")
        if value.ndim != 3:
            raise ValueError(f"{key} must be [B, N, D], got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{key} must contain finite values")
    if not (context.shape[0] == current.shape[0] == future.shape[0]):
        raise ValueError("context/current/future batch sizes must match")
    if not (context.shape[-1] == current.shape[-1] == future.shape[-1]):
        raise ValueError("context/current/future token dims must match")
    batch_size = int(context.shape[0])
    for key in ("sample_ids", "task_texts", "metadata"):
        value = shard[key]
        if not isinstance(value, list):
            raise TypeError(f"{key} must be a list")
        if len(value) != batch_size:
            raise ValueError(f"{key} length must match batch size {batch_size}")
    if strict:
        if not isinstance(shard["encoder_config"], dict):
            raise TypeError("encoder_config must be a dict")
        if not isinstance(shard["split"], str):
            raise TypeError("split must be a string")
    return True


def save_context_token_shard(path: str | Path, shard: dict[str, Any]) -> Path:
    validate_context_token_shard(shard, strict=True)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shard, out)
    return out


def load_context_token_shard(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    shard = torch.load(Path(path), map_location=map_location)
    validate_context_token_shard(shard, strict=True)
    return shard


def summarize_context_token_shard(shard: dict[str, Any]) -> dict[str, Any]:
    validate_context_token_shard(shard, strict=True)
    return {
        "schema_version": shard["schema_version"],
        "encoder_name": shard["encoder_name"],
        "num_samples": len(shard["sample_ids"]),
        "context_tokens_shape": list(shard["context_tokens"].shape),
        "current_tokens_shape": list(shard["current_tokens"].shape),
        "future_tokens_shape": list(shard["future_tokens"].shape),
        "split": shard["split"],
    }


class ContextTokenShardDataset(Dataset):
    """Load context/current/future token shards into memory."""

    def __init__(
        self,
        shard_paths: str | Path | Iterable[str | Path],
        shard_glob: str = "context_tokens_shard_*.pt",
        map_location: str | torch.device = "cpu",
        max_samples: int | None = None,
    ) -> None:
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.shard_paths = _resolve_paths(shard_paths, shard_glob)
        self.samples: list[dict[str, Any]] = []
        self.shard_summaries: list[dict[str, Any]] = []
        for shard_path in self.shard_paths:
            if max_samples is not None and len(self.samples) >= int(max_samples):
                break
            shard = load_context_token_shard(shard_path, map_location=map_location)
            self.shard_summaries.append(summarize_context_token_shard(shard))
            for row in range(int(shard["context_tokens"].shape[0])):
                if max_samples is not None and len(self.samples) >= int(max_samples):
                    break
                metadata = dict(shard["metadata"][row])
                metadata.setdefault("source_encoder", shard["encoder_name"])
                metadata.setdefault("token_shard_path", str(shard_path))
                self.samples.append(
                    {
                        "context_tokens": shard["context_tokens"][row].float().contiguous(),
                        "current_tokens": shard["current_tokens"][row].float().contiguous(),
                        "future_tokens": shard["future_tokens"][row].float().contiguous(),
                        "sample_id": str(shard["sample_ids"][row]),
                        "task_text": str(shard["task_texts"][row]),
                        "metadata": metadata,
                        "shard_path": str(shard_path),
                    }
                )
        if not self.samples:
            raise ValueError(f"No context token samples loaded from {self.shard_paths}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class ContextSelectorDataset(Dataset):
    """Pair context token shards with context-only predictive importance shards."""

    def __init__(
        self,
        token_shard_dir: str | Path | Iterable[str | Path],
        importance_shard_dir: str | Path | Iterable[str | Path],
        token_shard_glob: str = "context_tokens_shard_*.pt",
        importance_shard_glob: str = "importance_shard_*.pt",
        map_location: str | torch.device = "cpu",
        max_samples: int | None = None,
    ) -> None:
        token_paths = _resolve_paths(token_shard_dir, token_shard_glob)
        importance_paths = _resolve_paths(importance_shard_dir, importance_shard_glob)
        self.token_paths = token_paths
        self.importance_paths = importance_paths
        token_index: dict[str, dict[str, Any]] = {}
        for token_path in token_paths:
            shard = load_context_token_shard(token_path, map_location=map_location)
            for row in range(int(shard["context_tokens"].shape[0])):
                sample_id = str(shard["sample_ids"][row])
                if sample_id in token_index:
                    raise ValueError(f"Duplicate context sample_id: {sample_id}")
                metadata = dict(shard["metadata"][row])
                metadata["token_shard_path"] = str(token_path)
                token_index[sample_id] = {
                    "context_tokens": shard["context_tokens"][row].float().contiguous(),
                    "current_tokens": shard["current_tokens"][row].float().contiguous(),
                    "future_tokens": shard["future_tokens"][row].float().contiguous(),
                    "task_text": str(shard["task_texts"][row]),
                    "metadata": metadata,
                }
        self.samples: list[dict[str, Any]] = []
        seen: list[str] = []
        for importance_path in importance_paths:
            shard = load_importance_shard(importance_path, map_location=map_location)
            for row in range(int(shard["importance_scores"].shape[0])):
                sample_id = str(shard["sample_ids"][row])
                if sample_id not in token_index:
                    raise KeyError(f"Importance sample_id {sample_id!r} has no context token sample")
                token_sample = token_index[sample_id]
                importance = shard["importance_scores"][row].float().contiguous()
                importance_norm = shard["importance_scores_norm"][row].float().contiguous()
                if token_sample["context_tokens"].shape[0] != importance.shape[0]:
                    raise ValueError(f"context token/importance length mismatch for {sample_id}")
                metadata = dict(token_sample["metadata"])
                metadata["importance_shard_path"] = str(importance_path)
                metadata["importance_metadata"] = dict(shard["metadata"][row])
                self.samples.append(
                    {
                        **token_sample,
                        "sample_id": sample_id,
                        "metadata": metadata,
                        "importance_scores": importance,
                        "importance_scores_norm": importance_norm,
                    }
                )
                seen.append(sample_id)
                if max_samples is not None and len(self.samples) >= int(max_samples):
                    break
            if max_samples is not None and len(self.samples) >= int(max_samples):
                break
        if len(seen) != len(set(seen)):
            raise ValueError("Duplicate context importance sample_id found")
        if not self.samples:
            raise ValueError("No paired context selector samples were loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def context_token_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    return {
        "context_tokens": torch.stack([item["context_tokens"] for item in batch], dim=0),
        "current_tokens": torch.stack([item["current_tokens"] for item in batch], dim=0),
        "future_tokens": torch.stack([item["future_tokens"] for item in batch], dim=0),
        "sample_ids": [item["sample_id"] for item in batch],
        "task_texts": [item["task_text"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def context_selector_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated = context_token_collate_fn(batch)
    collated["importance_scores"] = torch.stack([item["importance_scores"] for item in batch], dim=0)
    collated["importance_scores_norm"] = torch.stack([item["importance_scores_norm"] for item in batch], dim=0)
    if collated["importance_scores"].shape != collated["context_tokens"].shape[:2]:
        raise ValueError("context importance must match [B, N_ctx]")
    return collated


def summarize_context_dataset(dataset: ContextTokenShardDataset | ContextSelectorDataset, split: str) -> dict[str, Any]:
    sample = dataset[0]
    return {
        "split": split,
        "num_samples": len(dataset),
        "num_context_tokens": int(sample["context_tokens"].shape[0]),
        "num_current_tokens": int(sample["current_tokens"].shape[0]),
        "num_future_tokens": int(sample["future_tokens"].shape[0]),
        "token_dim": int(sample["context_tokens"].shape[-1]),
        "context_token_shape": list(sample["context_tokens"].shape),
        "current_token_shape": list(sample["current_tokens"].shape),
        "future_token_shape": list(sample["future_tokens"].shape),
        "num_token_shards": len(getattr(dataset, "token_paths", getattr(dataset, "shard_paths", []))),
        "num_importance_shards": len(getattr(dataset, "importance_paths", [])),
    }
