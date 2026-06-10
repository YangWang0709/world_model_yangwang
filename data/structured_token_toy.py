"""Structured token toy data for Step 5.5 importance-signal diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from data.token_shards import TOKEN_SHARD_SCHEMA_VERSION, save_token_shard, utc_now_iso


@dataclass(frozen=True)
class StructuredTokenToyConfig:
    num_samples: int = 32
    num_tokens: int = 196
    token_dim: int = 768
    num_key_tokens: int = 4
    signal_scale: float = 3.0
    noise_scale: float = 0.05
    seed: int = 42
    split: str = "structured_toy"
    shard_size: int = 8


def _slot_signal(
    sample_index: int,
    slot_index: int,
    token_dim: int,
    num_key_tokens: int,
    signal_scale: float,
) -> torch.Tensor:
    signal = torch.zeros(token_dim, dtype=torch.float32)
    width = min(16, max(1, token_dim // max(1, num_key_tokens)))
    start = min(slot_index * width, max(0, token_dim - width))
    end = min(start + width, token_dim)
    base = torch.linspace(-1.0, 1.0, steps=end - start, dtype=torch.float32)
    sign = -1.0 if (sample_index + slot_index) % 2 else 1.0
    phase = (sample_index % 7) * 0.17 + slot_index * 0.31
    signal[start:end] = signal_scale * sign * torch.cos(base + phase)
    return signal


def generate_structured_token_sample(
    sample_index: int,
    config: StructuredTokenToyConfig,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Generate one sample where future target depends on a few key past tokens."""

    past_tokens = torch.randn(
        config.num_tokens,
        config.token_dim,
        generator=generator,
        dtype=torch.float32,
    ) * config.noise_scale
    future_tokens = torch.randn(
        config.num_tokens,
        config.token_dim,
        generator=generator,
        dtype=torch.float32,
    ) * config.noise_scale
    key_indices = torch.randperm(config.num_tokens, generator=generator)[: config.num_key_tokens].sort().values
    key_token_mask = torch.zeros(config.num_tokens, dtype=torch.float32)
    key_token_mask[key_indices] = 1.0

    target_global = torch.zeros(config.token_dim, dtype=torch.float32)
    for slot_index, token_index in enumerate(key_indices.tolist()):
        signal = _slot_signal(
            sample_index,
            slot_index,
            config.token_dim,
            config.num_key_tokens,
            config.signal_scale,
        )
        past_tokens[int(token_index)] += signal * float(config.num_tokens)
        target_global += signal
    target_global += torch.randn(config.token_dim, generator=generator) * config.noise_scale
    future_tokens += target_global.unsqueeze(0)

    key_index_list = [int(index) for index in key_indices.tolist()]
    key_mask_list = [int(value) for value in key_token_mask.to(torch.int64).tolist()]
    sample_id = f"structured_toy_{sample_index:06d}"
    return {
        "past_tokens": past_tokens,
        "future_tokens": future_tokens,
        "key_token_indices": key_index_list,
        "key_token_mask": key_token_mask,
        "sample_id": sample_id,
        "task_text": f"Predict the structured future signal for sample {sample_index}.",
        "metadata": {
            "sample_id": sample_id,
            "source": "structured_token_toy",
            "key_token_indices": key_index_list,
            "key_token_mask": key_mask_list,
            "key_token_mask_sum": int(key_token_mask.sum().item()),
            "signal_scale": float(config.signal_scale),
            "noise_scale": float(config.noise_scale),
            "num_tokens": int(config.num_tokens),
            "token_dim": int(config.token_dim),
        },
    }


def build_structured_token_shards(config: StructuredTokenToyConfig) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(config.seed)
    samples = [
        generate_structured_token_sample(index, config, generator)
        for index in range(config.num_samples)
    ]
    shards: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(samples), config.shard_size)):
        shard_samples = samples[start : start + config.shard_size]
        shard = {
            "schema_version": TOKEN_SHARD_SCHEMA_VERSION,
            "encoder_name": "structured_token_toy",
            "encoder_config": {
                "num_tokens": config.num_tokens,
                "token_dim": config.token_dim,
                "num_key_tokens": config.num_key_tokens,
                "signal_scale": config.signal_scale,
                "noise_scale": config.noise_scale,
                "seed": config.seed,
            },
            "created_at": utc_now_iso(),
            "split": config.split,
            "sample_ids": [sample["sample_id"] for sample in shard_samples],
            "task_texts": [sample["task_text"] for sample in shard_samples],
            "past_tokens": torch.stack([sample["past_tokens"] for sample in shard_samples], dim=0),
            "future_tokens": torch.stack([sample["future_tokens"] for sample in shard_samples], dim=0),
            "metadata": [sample["metadata"] for sample in shard_samples],
            "aux_labels": {
                "key_token_mask": torch.stack(
                    [sample["key_token_mask"] for sample in shard_samples],
                    dim=0,
                )
            },
        }
        shard["metadata"] = [
            {**metadata, "shard_index": shard_index}
            for metadata in shard["metadata"]
        ]
        shards.append(shard)
    return shards


def write_structured_token_shards(
    output_dir: str | Path,
    config: StructuredTokenToyConfig,
    overwrite: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    if output_path.exists() and overwrite:
        for path in output_path.glob("tokens_shard_*.pt"):
            path.unlink()
    output_path.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for shard_index, shard in enumerate(build_structured_token_shards(config)):
        shard_path = output_path / f"tokens_shard_{shard_index:06d}.pt"
        save_token_shard(shard_path, shard)
        written.append(str(shard_path))

    return {
        "output_dir": str(output_path),
        "num_samples": config.num_samples,
        "num_tokens": config.num_tokens,
        "token_dim": config.token_dim,
        "num_key_tokens": config.num_key_tokens,
        "signal_scale": config.signal_scale,
        "noise_scale": config.noise_scale,
        "seed": config.seed,
        "split": config.split,
        "shard_size": config.shard_size,
        "num_shards": len(written),
        "shard_files": [Path(path).name for path in written],
    }
