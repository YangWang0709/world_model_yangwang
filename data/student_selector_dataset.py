"""Paired token/importance dataset for Student selector training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from data.importance_shards import load_importance_shard
from data.token_shards import load_token_shard


DEFAULT_TOKEN_SHARD_DIR = Path("/home/ubuntu22/tgpawb_world_model/data/token_shards/structured_toy")
DEFAULT_IMPORTANCE_SHARD_DIR = Path(
    "/home/ubuntu22/tgpawb_world_model/data/importance_shards/structured_toy_teacher"
)


def _resolve_paths(paths: str | Path | Iterable[str | Path], shard_glob: str) -> list[Path]:
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        resolved = sorted(path.glob(shard_glob)) if path.is_dir() else [path]
    else:
        resolved = [Path(path) for path in paths]
    if not resolved:
        raise FileNotFoundError(f"No shard files found for {paths!r} with glob {shard_glob!r}")
    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Shard files do not exist: {missing}")
    return resolved


def _metadata_key_mask(metadata: dict[str, Any], num_tokens: int) -> torch.Tensor | None:
    if "key_token_mask" in metadata:
        mask = torch.as_tensor(metadata["key_token_mask"], dtype=torch.float32)
        if mask.shape != (num_tokens,):
            raise ValueError(
                f"metadata.key_token_mask must have shape [{num_tokens}], got {tuple(mask.shape)}"
            )
        return mask
    if "key_token_indices" in metadata:
        mask = torch.zeros(num_tokens, dtype=torch.float32)
        for index in metadata["key_token_indices"]:
            mask[int(index)] = 1.0
        return mask
    return None


class StudentSelectorDataset(Dataset):
    """Load token shards and predictive-importance shards aligned by sample_id."""

    def __init__(
        self,
        token_shard_dir: str | Path | Iterable[str | Path] = DEFAULT_TOKEN_SHARD_DIR,
        importance_shard_dir: str | Path | Iterable[str | Path] = DEFAULT_IMPORTANCE_SHARD_DIR,
        token_shard_glob: str = "tokens_shard_*.pt",
        importance_shard_glob: str = "importance_shard_*.pt",
        map_location: str = "cpu",
        require_key_token_mask: bool = True,
    ) -> None:
        self.token_paths = _resolve_paths(token_shard_dir, token_shard_glob)
        self.importance_paths = _resolve_paths(importance_shard_dir, importance_shard_glob)
        self.samples: list[dict[str, Any]] = []

        token_index: dict[str, dict[str, Any]] = {}
        for token_path in self.token_paths:
            shard = load_token_shard(token_path, map_location=map_location)
            aux_mask = shard.get("aux_labels", {}).get("key_token_mask")
            batch_size = int(shard["past_tokens"].shape[0])
            for row in range(batch_size):
                sample_id = str(shard["sample_ids"][row])
                if sample_id in token_index:
                    raise ValueError(f"Duplicate token sample_id: {sample_id}")
                num_tokens = int(shard["past_tokens"].shape[1])
                key_mask = None
                if isinstance(aux_mask, torch.Tensor):
                    key_mask = aux_mask[row].float().contiguous()
                if key_mask is None:
                    key_mask = _metadata_key_mask(shard["metadata"][row], num_tokens)
                token_index[sample_id] = {
                    "past_tokens": shard["past_tokens"][row].float().contiguous(),
                    "future_tokens": shard["future_tokens"][row].float().contiguous(),
                    "key_token_mask": key_mask,
                    "task_text": shard["task_texts"][row],
                    "metadata": dict(shard["metadata"][row]),
                    "token_shard_path": str(token_path),
                }

        paired_sample_ids: list[str] = []
        for importance_path in self.importance_paths:
            shard = load_importance_shard(importance_path, map_location=map_location)
            batch_size = int(shard["importance_scores"].shape[0])
            for row in range(batch_size):
                sample_id = str(shard["sample_ids"][row])
                if sample_id not in token_index:
                    raise KeyError(f"Importance sample_id {sample_id!r} has no matching token sample")
                token_sample = token_index[sample_id]
                key_mask = token_sample["key_token_mask"]
                if key_mask is None:
                    if require_key_token_mask:
                        raise KeyError(f"Sample {sample_id!r} is missing key_token_mask")

                importance_scores = shard["importance_scores"][row].float().contiguous()
                importance_scores_norm = shard["importance_scores_norm"][row].float().contiguous()
                if token_sample["past_tokens"].shape[0] != importance_scores.shape[0]:
                    raise ValueError(
                        f"Token/importance length mismatch for {sample_id}: "
                        f"{token_sample['past_tokens'].shape[0]} vs {importance_scores.shape[0]}"
                    )
                metadata = dict(token_sample["metadata"])
                metadata["importance_metadata"] = dict(shard["metadata"][row])
                metadata["importance_shard_path"] = str(importance_path)
                self.samples.append(
                    {
                        "past_tokens": token_sample["past_tokens"],
                        "future_tokens": token_sample["future_tokens"],
                        "importance_scores": importance_scores,
                        "importance_scores_norm": importance_scores_norm,
                        "key_token_mask": key_mask.float().contiguous() if key_mask is not None else None,
                        "has_key_token_mask": key_mask is not None,
                        "sample_id": sample_id,
                        "task_text": token_sample["task_text"],
                        "metadata": metadata,
                    }
                )
                paired_sample_ids.append(sample_id)

        if len(paired_sample_ids) != len(set(paired_sample_ids)):
            raise ValueError("Duplicate paired sample_id found in importance shards")
        missing_importance = sorted(set(token_index).difference(paired_sample_ids))
        if missing_importance:
            raise KeyError(f"Token samples missing importance labels: {missing_importance[:5]}")
        if not self.samples:
            raise ValueError("No paired Student selector samples were loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.samples):
            raise IndexError(index)
        return self.samples[index]


def student_selector_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    first_future_ndim = batch[0]["future_tokens"].ndim
    if any(item["future_tokens"].ndim != first_future_ndim for item in batch):
        raise ValueError("All future_tokens in a batch must have the same rank")
    past_tokens = torch.stack([item["past_tokens"] for item in batch], dim=0)
    future_tokens = torch.stack([item["future_tokens"] for item in batch], dim=0)
    importance_scores = torch.stack([item["importance_scores"] for item in batch], dim=0)
    importance_scores_norm = torch.stack([item["importance_scores_norm"] for item in batch], dim=0)
    has_key_token_mask = all(bool(item.get("has_key_token_mask", False)) for item in batch)
    key_token_mask = (
        torch.stack([item["key_token_mask"] for item in batch], dim=0)
        if has_key_token_mask
        else None
    )
    if past_tokens.ndim != 3:
        raise ValueError(f"past_tokens must collate to [B, N, D], got {tuple(past_tokens.shape)}")
    if future_tokens.ndim not in (2, 3):
        raise ValueError(f"future_tokens must collate to [B, D] or [B, N, D], got {tuple(future_tokens.shape)}")
    if importance_scores.shape != past_tokens.shape[:2]:
        raise ValueError("importance_scores must have shape [B, N] matching past_tokens")
    if importance_scores_norm.shape != past_tokens.shape[:2]:
        raise ValueError("importance_scores_norm must have shape [B, N] matching past_tokens")
    if key_token_mask is not None and key_token_mask.shape != past_tokens.shape[:2]:
        raise ValueError("key_token_mask must have shape [B, N] matching past_tokens")
    return {
        "past_tokens": past_tokens,
        "future_tokens": future_tokens,
        "importance_scores": importance_scores,
        "importance_scores_norm": importance_scores_norm,
        "key_token_mask": key_token_mask,
        "has_key_token_mask": has_key_token_mask,
        "sample_ids": [item["sample_id"] for item in batch],
        "task_texts": [item["task_text"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
