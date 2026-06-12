"""Deterministic train/val splitting for Step29 selected BridgeData windows."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from data.bridgedata_v2_tfds_window_sampler import read_window_manifest_jsonl


def build_train_val_split(
    windows: list[dict[str, Any]],
    train_ratio: float = 0.75,
    min_train_windows: int = 24,
    min_val_windows: int = 8,
) -> dict[str, Any]:
    if not windows:
        raise ValueError("cannot split an empty window list")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in windows:
        groups.setdefault(str(record["trajectory_id"]), []).append(record)
    for records in groups.values():
        records.sort(key=lambda item: str(item["sample_id"]))

    total = len(windows)
    target_train = round(total * float(train_ratio))
    target_val = max(int(min_val_windows), total - int(target_train))
    target_val = min(target_val, total - 1)

    split_type = "trajectory_group_if_possible"
    val_trajectories = _choose_val_trajectories(groups, target_val, min_train_windows)
    if not val_trajectories:
        split_type = "window_fallback"
        ordered = sorted(windows, key=lambda item: str(item["sample_id"]))
        val_records = ordered[-target_val:]
        train_records = ordered[:-target_val]
    else:
        val_set = set(val_trajectories)
        val_records = [record for trajectory_id in val_trajectories for record in groups[trajectory_id]]
        train_records = [
            record
            for trajectory_id in sorted(groups)
            if trajectory_id not in val_set
            for record in groups[trajectory_id]
        ]

    train_ids = [str(record["sample_id"]) for record in train_records]
    val_ids = [str(record["sample_id"]) for record in val_records]
    train_traj = sorted({str(record["trajectory_id"]) for record in train_records})
    val_traj = sorted({str(record["trajectory_id"]) for record in val_records})
    duplicates = sorted(set(train_ids) & set(val_ids))
    return {
        "split_type": split_type,
        "num_selected_windows": total,
        "num_train_windows": len(train_records),
        "num_val_windows": len(val_records),
        "train_sample_ids": train_ids,
        "val_sample_ids": val_ids,
        "train_trajectories": train_traj,
        "val_trajectories": val_traj,
        "train_trajectory_count": len(train_traj),
        "val_trajectory_count": len(val_traj),
        "train_val_trajectory_disjoint": not bool(set(train_traj) & set(val_traj)),
        "duplicate_sample_ids_between_train_val": bool(duplicates),
        "duplicate_sample_ids": duplicates,
        "train_ratio": float(train_ratio),
        "val_ratio": len(val_records) / total,
        "deterministic": True,
    }


def split_selected_windows_from_config(config: dict[str, Any]) -> dict[str, Any]:
    windows = read_window_manifest_jsonl(config["output"]["selected_windows_jsonl"])
    split_cfg = config.get("split", {})
    split = build_train_val_split(
        windows,
        train_ratio=float(split_cfg.get("train_ratio", 0.75)),
        min_train_windows=int(split_cfg.get("min_train_windows", 24)),
        min_val_windows=int(split_cfg.get("min_val_windows", 8)),
    )
    path = Path(config["output"]["split_json"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")
    return split


def _choose_val_trajectories(
    groups: dict[str, list[dict[str, Any]]],
    target_val: int,
    min_train_windows: int,
) -> list[str]:
    trajectory_ids = sorted(groups)
    if len(trajectory_ids) < 2:
        return []
    total = sum(len(records) for records in groups.values())
    best: tuple[int, int, int, tuple[str, ...]] | None = None
    for size in range(1, len(trajectory_ids)):
        for combo in itertools.combinations(trajectory_ids, size):
            val_count = sum(len(groups[trajectory_id]) for trajectory_id in combo)
            train_count = total - val_count
            if val_count < 1 or train_count < 1:
                continue
            penalty = 0
            if train_count < min_train_windows:
                penalty += min_train_windows - train_count
            score = (penalty, abs(val_count - target_val), size, combo)
            if best is None or score < best:
                best = score
    return list(best[3]) if best is not None else []

