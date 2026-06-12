"""Multi-seed train/val splits for Step30A selected BridgeData windows."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any

from data.bridgedata_v2_tfds_window_sampler import read_window_manifest_jsonl


def build_multiseed_splits(
    windows: list[dict[str, Any]],
    seeds: list[int] | tuple[int, ...] = (42, 123, 999),
    train_ratio: float = 0.75,
    val_ratio: float = 0.25,
    min_val_windows_by_count: dict[int, int] | None = None,
) -> dict[str, Any]:
    if not windows:
        raise ValueError("cannot split an empty window list")
    total = len(windows)
    min_val_windows_by_count = min_val_windows_by_count or {}
    min_val = int(min_val_windows_by_count.get(total, max(1, round(total * val_ratio))))
    target_val = max(min_val, round(total * float(val_ratio)))

    groups = _group_by_trajectory(windows)
    splits = [
        _build_one_split(
            windows=windows,
            groups=groups,
            seed=int(seed),
            train_ratio=float(train_ratio),
            target_val=int(target_val),
        )
        for seed in seeds
    ]
    return {
        "stage": "bridgedata_v2_tfds_window_diversity_step30a",
        "num_selected_windows": total,
        "num_selected_trajectories": len(groups),
        "seeds": [int(seed) for seed in seeds],
        "num_splits": len(splits),
        "splits": splits,
        "all_duplicate_sample_ids_between_train_val": any(
            bool(split["duplicate_sample_ids_between_train_val"]) for split in splits
        ),
        "all_train_val_trajectory_disjoint": all(bool(split["train_val_trajectory_disjoint"]) for split in splits),
        "deterministic": True,
    }


def write_multiseed_splits_from_manifest(
    selected_windows_jsonl: str | Path,
    output_multiseed_splits_json: str | Path,
    seeds: list[int] | tuple[int, ...] = (42, 123, 999),
    train_ratio: float = 0.75,
    val_ratio: float = 0.25,
    min_val_windows_by_count: dict[int, int] | None = None,
) -> dict[str, Any]:
    windows = read_window_manifest_jsonl(selected_windows_jsonl)
    payload = build_multiseed_splits(
        windows,
        seeds=seeds,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        min_val_windows_by_count=min_val_windows_by_count,
    )
    output = Path(output_multiseed_splits_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _build_one_split(
    windows: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
    seed: int,
    train_ratio: float,
    target_val: int,
) -> dict[str, Any]:
    val_trajectories, fallback_reason = _choose_val_trajectories(groups, target_val, seed)
    if val_trajectories:
        val_set = set(val_trajectories)
        val_records = [record for trajectory_id in val_trajectories for record in groups[trajectory_id]]
        train_records = [
            record
            for trajectory_id in sorted(groups)
            if trajectory_id not in val_set
            for record in groups[trajectory_id]
        ]
        split_type = "trajectory_disjoint"
    else:
        split_type = "trajectory_group_aware_window_fallback"
        fallback_reason = fallback_reason or "fewer than two usable trajectories"
        ordered = sorted(windows, key=lambda item: (str(item["trajectory_id"]), str(item["sample_id"])))
        rng = random.Random(seed)
        shuffled = ordered[:]
        rng.shuffle(shuffled)
        val_count = min(max(1, target_val), len(shuffled) - 1)
        val_records = sorted(shuffled[:val_count], key=lambda item: str(item["sample_id"]))
        train_records = sorted(shuffled[val_count:], key=lambda item: str(item["sample_id"]))

    train_ids = [str(record["sample_id"]) for record in train_records]
    val_ids = [str(record["sample_id"]) for record in val_records]
    train_traj = sorted({str(record["trajectory_id"]) for record in train_records})
    val_traj = sorted({str(record["trajectory_id"]) for record in val_records})
    duplicates = sorted(set(train_ids) & set(val_ids))
    return {
        "split_seed": int(seed),
        "split_type": split_type,
        "fallback_used": split_type != "trajectory_disjoint",
        "fallback_reason": None if split_type == "trajectory_disjoint" else fallback_reason,
        "num_selected_windows": len(windows),
        "num_train_windows": len(train_records),
        "num_val_windows": len(val_records),
        "train_ratio": float(train_ratio),
        "val_ratio": len(val_records) / len(windows),
        "train_sample_ids": train_ids,
        "val_sample_ids": val_ids,
        "train_trajectories": train_traj,
        "val_trajectories": val_traj,
        "train_trajectory_count": len(train_traj),
        "val_trajectory_count": len(val_traj),
        "train_val_trajectory_disjoint": not bool(set(train_traj) & set(val_traj)),
        "duplicate_sample_ids_between_train_val": bool(duplicates),
        "duplicate_sample_ids": duplicates,
        "deterministic": True,
    }


def _choose_val_trajectories(
    groups: dict[str, list[dict[str, Any]]],
    target_val: int,
    seed: int,
) -> tuple[list[str], str | None]:
    trajectory_ids = sorted(groups)
    if len(trajectory_ids) < 2:
        return [], "trajectory-disjoint split requires at least two trajectories"
    total = sum(len(records) for records in groups.values())
    rng = random.Random(seed)
    jitter = {trajectory_id: rng.random() for trajectory_id in trajectory_ids}
    best: tuple[int, int, float, tuple[str, ...]] | None = None
    for size in range(1, len(trajectory_ids)):
        for combo in itertools.combinations(trajectory_ids, size):
            val_count = sum(len(groups[trajectory_id]) for trajectory_id in combo)
            train_count = total - val_count
            if val_count < 1 or train_count < 1:
                continue
            score = (
                abs(val_count - target_val),
                abs(round(total * 0.75) - train_count),
                sum(jitter[trajectory_id] for trajectory_id in combo),
                combo,
            )
            if best is None or score < best:
                best = score
    if best is None:
        return [], "no non-empty train/val trajectory grouping was possible"
    return list(best[3]), None


def _group_by_trajectory(windows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in windows:
        groups.setdefault(str(record["trajectory_id"]), []).append(dict(record))
    for records in groups.values():
        records.sort(key=lambda item: str(item["sample_id"]))
    return dict(sorted(groups.items()))
