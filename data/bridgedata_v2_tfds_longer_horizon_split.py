"""Trajectory-aware multi-seed splits for Step32 longer-horizon windows."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from data.bridgedata_v2_tfds_longer_horizon_manifest import STAGE, read_jsonl


def write_longer_horizon_splits(
    manifest_jsonl: str | Path,
    output_json: str | Path,
    *,
    seeds: list[int],
    train_ratio: float = 0.75,
    val_ratio: float = 0.25,
    trajectory_disjoint_if_possible: bool = True,
) -> dict[str, Any]:
    records = read_jsonl(manifest_jsonl)
    by_gap: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_gap[int(record["horizon_gap"])].append(record)

    splits: list[dict[str, Any]] = []
    for gap, gap_records in sorted(by_gap.items()):
        for seed in seeds:
            splits.append(
                build_split_for_horizon(
                    gap_records,
                    horizon_gap=gap,
                    seed=int(seed),
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    trajectory_disjoint_if_possible=trajectory_disjoint_if_possible,
                )
            )

    payload = {
        "stage": STAGE,
        "safe_stop": False,
        "num_splits": len(splits),
        "seeds": [int(seed) for seed in seeds],
        "split_seeds": [int(seed) for seed in seeds],
        "horizons": sorted(by_gap),
        "splits": splits,
        "sample_ids_unique_per_split": all(_ids_disjoint(split) for split in splits),
        "safety_gate_pass": True,
    }
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_split_for_horizon(
    records: list[dict[str, Any]],
    *,
    horizon_gap: int,
    seed: int,
    train_ratio: float = 0.75,
    val_ratio: float = 0.25,
    trajectory_disjoint_if_possible: bool = True,
) -> dict[str, Any]:
    del train_ratio
    if not records:
        return _empty_split(horizon_gap, seed, "no records for horizon")
    desired_val = max(1, int(round(len(records) * float(val_ratio))))
    if trajectory_disjoint_if_possible:
        split = _trajectory_disjoint_split(records, horizon_gap, seed, desired_val)
        if split is not None:
            return split
    return _sample_level_split(records, horizon_gap, seed, desired_val, "trajectory-disjoint split unavailable")


def _trajectory_disjoint_split(
    records: list[dict[str, Any]],
    horizon_gap: int,
    seed: int,
    desired_val: int,
) -> dict[str, Any] | None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["trajectory_id"])].append(record)
    if len(grouped) < 2:
        return None
    trajectory_ids = sorted(grouped)
    rng = random.Random(seed + int(horizon_gap) * 1009)
    rng.shuffle(trajectory_ids)
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
    return _split_payload(
        horizon_gap,
        seed,
        train_records,
        val_records,
        trajectory_disjoint=True,
        fallback_used=False,
        fallback_reason=None,
    )


def _sample_level_split(
    records: list[dict[str, Any]],
    horizon_gap: int,
    seed: int,
    desired_val: int,
    reason: str,
) -> dict[str, Any]:
    ordered = list(records)
    random.Random(seed + int(horizon_gap) * 101).shuffle(ordered)
    val_records = ordered[:desired_val]
    train_records = ordered[desired_val:]
    return _split_payload(
        horizon_gap,
        seed,
        train_records,
        val_records,
        trajectory_disjoint=False,
        fallback_used=True,
        fallback_reason=reason,
    )


def _split_payload(
    horizon_gap: int,
    seed: int,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    *,
    trajectory_disjoint: bool,
    fallback_used: bool,
    fallback_reason: str | None,
) -> dict[str, Any]:
    return {
        "horizon_gap": int(horizon_gap),
        "split_seed": int(seed),
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


def _empty_split(horizon_gap: int, seed: int, reason: str) -> dict[str, Any]:
    return _split_payload(
        horizon_gap,
        seed,
        [],
        [],
        trajectory_disjoint=False,
        fallback_used=True,
        fallback_reason=reason,
    )


def _ids_disjoint(split: dict[str, Any]) -> bool:
    train = set(split.get("train_sample_ids", []))
    val = set(split.get("val_sample_ids", []))
    return not (train & val)
