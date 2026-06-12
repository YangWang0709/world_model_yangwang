"""Shard-aware split helpers for Step33B data-diversity diagnostics."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from data.bridgedata_v2_tfds_longer_horizon_manifest import read_jsonl


STAGE = "bridgedata_v2_tfds_data_diversity_step33b"


def build_shard_split(
    records: list[dict[str, Any]],
    *,
    split_seed: int,
    train_ratio: float = 0.75,
    val_ratio: float = 0.25,
    trajectory_disjoint_if_possible: bool = True,
    shard_name: str = "shard1",
) -> dict[str, Any]:
    del train_ratio
    if not records:
        return _split_payload([], [], split_seed, shard_name, False, True, "no records")
    desired_val = max(1, int(round(len(records) * float(val_ratio))))
    if trajectory_disjoint_if_possible:
        split = _trajectory_disjoint_split(records, split_seed, desired_val, shard_name)
        if split is not None:
            return split
    ordered = list(records)
    random.Random(int(split_seed)).shuffle(ordered)
    return _split_payload(
        ordered[desired_val:],
        ordered[:desired_val],
        split_seed,
        shard_name,
        False,
        True,
        "trajectory-disjoint split unavailable",
    )


def write_step33b_splits(
    shard1_manifest_jsonl: str | Path,
    output_json: str | Path,
    *,
    seeds: list[int],
    train_ratio: float = 0.75,
    val_ratio: float = 0.25,
    trajectory_disjoint_if_possible: bool = True,
) -> dict[str, Any]:
    shard1_records = read_jsonl(shard1_manifest_jsonl)
    splits = [
        build_shard_split(
            shard1_records,
            split_seed=int(seed),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            trajectory_disjoint_if_possible=trajectory_disjoint_if_possible,
            shard_name="shard1",
        )
        for seed in seeds
    ]
    payload = {
        "stage": STAGE,
        "safe_stop": False,
        "split_seeds": [int(seed) for seed in seeds],
        "splits": splits,
        "num_splits": len(splits),
        "sample_ids_unique_per_split": all(_ids_disjoint(split) for split in splits),
        "trajectory_disjoint_if_possible": bool(trajectory_disjoint_if_possible),
        "safety_gate_pass": True,
    }
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _trajectory_disjoint_split(
    records: list[dict[str, Any]],
    split_seed: int,
    desired_val: int,
    shard_name: str,
) -> dict[str, Any] | None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["trajectory_id"])].append(record)
    if len(grouped) < 2:
        return None
    trajectory_ids = sorted(grouped)
    random.Random(int(split_seed)).shuffle(trajectory_ids)
    val_trajectories: list[str] = []
    val_records: list[dict[str, Any]] = []
    for trajectory_id in trajectory_ids:
        if len(val_trajectories) >= len(trajectory_ids) - 1:
            break
        val_trajectories.append(trajectory_id)
        val_records.extend(grouped[trajectory_id])
        if len(val_records) >= desired_val:
            break
    train_records = [record for record in records if str(record["trajectory_id"]) not in set(val_trajectories)]
    if not train_records or not val_records:
        return None
    return _split_payload(train_records, val_records, split_seed, shard_name, True, False, None)


def _split_payload(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    split_seed: int,
    shard_name: str,
    trajectory_disjoint: bool,
    fallback_used: bool,
    fallback_reason: str | None,
) -> dict[str, Any]:
    return {
        "split_seed": int(split_seed),
        "shard": shard_name,
        "horizon_gap": 0,
        "train_sample_ids": [str(record["sample_id"]) for record in train_records],
        "val_sample_ids": [str(record["sample_id"]) for record in val_records],
        "num_train_windows": len(train_records),
        "num_val_windows": len(val_records),
        "train_trajectories": sorted({str(record["trajectory_id"]) for record in train_records}),
        "val_trajectories": sorted({str(record["trajectory_id"]) for record in val_records}),
        "train_val_trajectory_disjoint": bool(trajectory_disjoint),
        "trajectory_disjoint": bool(trajectory_disjoint),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
    }


def _ids_disjoint(split: dict[str, Any]) -> bool:
    return not (set(split.get("train_sample_ids", [])) & set(split.get("val_sample_ids", [])))
