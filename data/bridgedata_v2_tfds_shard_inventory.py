"""Local shard inventory helpers for Step33B BridgeData TFDS diversity checks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


STAGE = "bridgedata_v2_tfds_shard_inventory_step33b"
SHARD_RE = re.compile(r"bridge_dataset-train\.tfrecord-(?P<index>\d{5})-of-(?P<total>\d{5})$")


def read_shard_lengths(dataset_root: str | Path) -> list[int]:
    info_path = Path(dataset_root) / "dataset_info.json"
    if not info_path.exists():
        return []
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for split in info.get("splits", []):
        if split.get("name") == "train":
            return [int(item) for item in split.get("shardLengths", [])]
    return []


def shard_episode_range(shard_lengths: list[int], shard_index: int) -> tuple[int, int]:
    if shard_index < 0 or shard_index >= len(shard_lengths):
        raise ValueError(f"shard_index out of range: {shard_index}")
    start = sum(int(value) for value in shard_lengths[:shard_index])
    end = start + int(shard_lengths[shard_index])
    return start, end


def inventory_local_tfds_shards(
    dataset_root: str | Path,
    *,
    selected_shard_index: int | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root)
    shard_lengths = read_shard_lengths(root)
    shards: list[dict[str, Any]] = []
    for path in sorted(root.glob("bridge_dataset-train.tfrecord-*-of-*")):
        match = SHARD_RE.match(path.name)
        if not match:
            continue
        index = int(match.group("index"))
        episode_start = None
        episode_end = None
        num_examples = None
        if shard_lengths and index < len(shard_lengths):
            episode_start, episode_end = shard_episode_range(shard_lengths, index)
            num_examples = shard_lengths[index]
        shards.append(
            {
                "shard_index": index,
                "filename": path.name,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": int(path.stat().st_size),
                "size_mb": path.stat().st_size / (1024**2),
                "num_examples": num_examples,
                "episode_start": episode_start,
                "episode_end_exclusive": episode_end,
                "is_selected_second_shard": selected_shard_index is not None and index == int(selected_shard_index),
            }
        )
    payload = {
        "stage": STAGE,
        "dataset_root": str(root),
        "dataset_root_exists": root.exists(),
        "num_local_train_shards": len(shards),
        "local_shard_indices": [int(item["shard_index"]) for item in shards],
        "selected_shard_index": selected_shard_index,
        "selected_shard_present": any(
            selected_shard_index is not None and int(item["shard_index"]) == int(selected_shard_index)
            for item in shards
        ),
        "shard_lengths_available": bool(shard_lengths),
        "num_shard_lengths": len(shard_lengths),
        "known_examples_per_available_shard_mean": mean(
            [int(item["num_examples"]) for item in shards if item.get("num_examples") is not None]
        )
        if any(item.get("num_examples") is not None for item in shards)
        else None,
        "shards": shards,
        "raw_zip_downloaded": False,
        "full_tfds_downloaded": False,
        "model_download_performed": False,
        "safety_gate_pass": True,
    }
    return payload


def write_shard_inventory(summary: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
