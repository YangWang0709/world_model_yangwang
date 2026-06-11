"""Dataset for Step 3 token shard files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from .token_shards import load_token_shard, summarize_token_shard, validate_token_shard


DEFAULT_TOKEN_SHARD_DIR = Path("/home/ubuntu22/tgpawb_world_model/data/token_shards/dummy_toy")


def _resolve_shard_paths(paths: str | Path | Iterable[str | Path], shard_glob: str) -> list[Path]:
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        if path.is_dir():
            resolved = sorted(path.glob(shard_glob))
        else:
            resolved = [path]
    else:
        resolved = [Path(path) for path in paths]

    if not resolved:
        raise FileNotFoundError(f"No token shard files found for {paths!r} with glob {shard_glob!r}")

    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Token shard files do not exist: {missing}")

    return resolved


class TokenShardDataset(Dataset):
    """Load small token shards into memory for tiny teacher sanity training."""

    def __init__(
        self,
        shard_paths: str | Path | Iterable[str | Path] = DEFAULT_TOKEN_SHARD_DIR,
        shard_glob: str = "tokens_shard_*.pt",
        map_location: str = "cpu",
        max_samples: int | None = None,
    ) -> None:
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        self.shard_paths = _resolve_shard_paths(shard_paths, shard_glob)
        self.samples: list[dict[str, Any]] = []
        self.shard_summaries: list[dict[str, Any]] = []

        for shard_path in self.shard_paths:
            if max_samples is not None and len(self.samples) >= max_samples:
                break
            shard = load_token_shard(shard_path, map_location=map_location)
            validate_token_shard(shard, strict=True)
            self.shard_summaries.append(summarize_token_shard(shard))

            batch_size = shard["past_tokens"].shape[0]
            for index in range(batch_size):
                if max_samples is not None and len(self.samples) >= max_samples:
                    break
                past_tokens = shard["past_tokens"][index]
                future_tokens = shard["future_tokens"][index]
                if past_tokens.ndim != 2:
                    raise ValueError(
                        f"past token sample must have shape [N, D], got {tuple(past_tokens.shape)}"
                    )
                if future_tokens.ndim not in (1, 2):
                    raise ValueError(
                        "future token sample must have shape [D] or [N, D], "
                        f"got {tuple(future_tokens.shape)}"
                    )

                self.samples.append(
                    {
                        "past_tokens": past_tokens.float().contiguous(),
                        "future_tokens": future_tokens.float().contiguous(),
                        "sample_id": shard["sample_ids"][index],
                        "task_text": shard["task_texts"][index],
                        "metadata": shard["metadata"][index],
                        "shard_path": str(shard_path),
                    }
                )

        if not self.samples:
            raise ValueError(f"No samples loaded from token shards: {self.shard_paths}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.samples[index]


def token_shard_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate token shard samples with strict future target rank checks."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")

    past_tokens = torch.stack([item["past_tokens"] for item in batch], dim=0)
    first_future_ndim = batch[0]["future_tokens"].ndim
    if any(item["future_tokens"].ndim != first_future_ndim for item in batch):
        raise ValueError("All future_tokens in a batch must have the same rank")
    future_tokens = torch.stack([item["future_tokens"] for item in batch], dim=0)

    if past_tokens.ndim != 3:
        raise ValueError(f"Collated past_tokens must be [B, N, D], got {tuple(past_tokens.shape)}")
    if future_tokens.ndim not in (2, 3):
        raise ValueError(
            f"Collated future_tokens must be [B, D] or [B, N, D], got {tuple(future_tokens.shape)}"
        )

    return {
        "past_tokens": past_tokens,
        "future_tokens": future_tokens,
        "sample_ids": [item["sample_id"] for item in batch],
        "task_texts": [item["task_text"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
        "shard_paths": [item["shard_path"] for item in batch],
    }
