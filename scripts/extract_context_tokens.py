"""Extract frozen VideoMAE tokens for BAIR context/current/future windows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bair_context_window_dataset import BAIRContextWindowDataset
from data.context_token_shard_dataset import save_context_token_shard, summarize_context_token_shard, utc_now_iso
from encoders.frozen_video_encoder import build_frozen_video_encoder


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def resolve_device(name: str | None) -> torch.device:
    if name is None or name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "context_video": torch.stack([item["context_video"] for item in batch], dim=0),
        "current_video": torch.stack([item["current_video"] for item in batch], dim=0),
        "future_video": torch.stack([item["future_video"] for item in batch], dim=0),
        "sample_ids": [item["sample_id"] for item in batch],
        "task_texts": [item["task_text"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def _encoder_cfg(config: dict[str, Any], num_frames: int, device: torch.device) -> dict[str, Any]:
    cfg = dict(config["encoder"])
    cfg["num_frames"] = int(num_frames)
    cfg["device"] = str(device)
    return cfg


def _build_dataset(config: dict[str, Any], split: str) -> BAIRContextWindowDataset:
    data_cfg = config["dataset"]
    max_samples = data_cfg.get(f"max_{split}_samples")
    return BAIRContextWindowDataset(
        root=data_cfg["root"],
        split=split,
        metadata_file=data_cfg.get(f"{split}_metadata_file"),
        max_samples=int(max_samples) if max_samples is not None else None,
    )


def _extract_split(
    *,
    config: dict[str, Any],
    split: str,
    out_dir: Path,
    context_encoder: Any,
    short_encoder: Any,
    overwrite: bool,
    start_time: float,
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.glob("context_tokens_shard_*.pt")) and not overwrite:
        summary_path = out_dir / "extraction_summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"Output dir already has context token shards: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old in out_dir.glob("context_tokens_shard_*.pt"):
            old.unlink()
    extraction_cfg = config["extraction"]
    dataset = _build_dataset(config, split)
    loader = DataLoader(
        dataset,
        batch_size=int(extraction_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(extraction_cfg.get("num_workers", 0)),
        collate_fn=_collate,
    )
    buffer: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    shard_index = 0
    shard_size = int(extraction_cfg.get("shard_size", 4))

    def flush() -> None:
        nonlocal buffer, shard_index
        if not buffer:
            return
        shard = {
            "schema_version": "0.1.0",
            "encoder_name": "videomae",
            "encoder_config": {
                "requested_encoder": "videomae",
                "actual_encoder": "videomae",
                "used_fallback": False,
                "model_name_or_path": config["encoder"].get("model_name_or_path"),
                "context_num_frames": int(config["encoder"].get("context_num_frames", 8)),
                "short_num_frames": int(config["encoder"].get("short_num_frames", 4)),
                "context_runtime_summary": getattr(context_encoder.impl, "last_encode_summary", {}),
                "short_runtime_summary": getattr(short_encoder.impl, "last_encode_summary", {}),
            },
            "created_at": utc_now_iso(),
            "split": split,
            "sample_ids": [sid for item in buffer for sid in item["sample_ids"]],
            "task_texts": [text for item in buffer for text in item["task_texts"]],
            "context_tokens": torch.cat([item["context_tokens"] for item in buffer], dim=0).cpu(),
            "current_tokens": torch.cat([item["current_tokens"] for item in buffer], dim=0).cpu(),
            "future_tokens": torch.cat([item["future_tokens"] for item in buffer], dim=0).cpu(),
            "metadata": [meta for item in buffer for meta in item["metadata"]],
        }
        path = out_dir / f"context_tokens_shard_{shard_index:06d}.pt"
        save_context_token_shard(path, shard)
        summary = summarize_context_token_shard(shard)
        summary["path"] = str(path)
        shard_summaries.append(summary)
        print(f"WROTE_CONTEXT_TOKEN_SHARD {path} {summary}", flush=True)
        shard_index += 1
        buffer = []

    with torch.no_grad():
        for batch in loader:
            buffer.append(
                {
                    "context_tokens": context_encoder.encode(batch["context_video"]),
                    "current_tokens": short_encoder.encode(batch["current_video"]),
                    "future_tokens": short_encoder.encode(batch["future_video"]),
                    "sample_ids": batch["sample_ids"],
                    "task_texts": batch["task_texts"],
                    "metadata": batch["metadata"],
                }
            )
            if sum(int(item["context_tokens"].shape[0]) for item in buffer) >= shard_size:
                flush()
    flush()
    summary = {
        "split": split,
        "dataset_size": len(dataset),
        "num_shards": len(shard_summaries),
        "output_dir": str(out_dir),
        "shards": shard_summaries,
        "context_token_shape": shard_summaries[0]["context_tokens_shape"] if shard_summaries else None,
        "current_token_shape": shard_summaries[0]["current_tokens_shape"] if shard_summaries else None,
        "future_token_shape": shard_summaries[0]["future_tokens_shape"] if shard_summaries else None,
        "used_fallback": False,
        "elapsed_time_sec": round(time.perf_counter() - start_time, 3),
    }
    (out_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def extract_context_tokens_from_config(config_path: str | Path, overwrite: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    start = time.perf_counter()
    device = resolve_device(config["extraction"].get("device"))
    context_encoder = build_frozen_video_encoder(_encoder_cfg(config, int(config["encoder"].get("context_num_frames", 8)), device))
    short_encoder = build_frozen_video_encoder(_encoder_cfg(config, int(config["encoder"].get("short_num_frames", 4)), device))
    if not context_encoder.is_available() or not short_encoder.is_available():
        raise RuntimeError(f"VideoMAE unavailable; context={context_encoder.availability}, short={short_encoder.availability}")
    output_root = Path(config["extraction"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    split_summaries = {}
    for split in config["dataset"].get("split_names", ["train", "test"]):
        out_dir = Path(config["extraction"].get(f"{split}_output_dir", output_root / split))
        split_summaries[str(split)] = _extract_split(
            config=config,
            split=str(split),
            out_dir=out_dir,
            context_encoder=context_encoder,
            short_encoder=short_encoder,
            overwrite=overwrite,
            start_time=start,
        )
    summary = {
        "config": str(config_path),
        "dataset_name": config["dataset"]["name"],
        "output_root": str(output_root),
        "split_summaries": split_summaries,
        "total_samples": sum(int(item["dataset_size"]) for item in split_summaries.values()),
        "total_shards": sum(int(item["num_shards"]) for item in split_summaries.values()),
        "used_fallback": False,
        "elapsed_time_sec": round(time.perf_counter() - start, 3),
    }
    (output_root / "extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "token_extraction_bair_context_videomae_1000_128.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_context_tokens_from_config(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
