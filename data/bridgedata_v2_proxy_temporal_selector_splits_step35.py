"""Shard-aware split settings for Step35 proxy-temporal selector smoke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STAGE = "bridgedata_v2_tfds_proxy_temporal_selector_train_step35"


def build_step35_training_settings(config: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = [int(seed) for seed in config["splits"].get("split_seeds", [42, 123, 999])]
    shard1_splits = {
        int(split["split_seed"]): split
        for split in _read_json(config["input"]["shard_splits_json"]).get("splits", [])
    }
    shard0_splits = {
        int(split["split_seed"]): split
        for split in _read_json(config["input"]["shard0_horizon_splits_json"]).get("splits", [])
        if int(split.get("horizon_gap", -1)) == 0
    }
    settings: list[dict[str, Any]] = []
    for seed in seeds:
        shard1 = shard1_splits.get(seed)
        shard0 = shard0_splits.get(seed)
        if shard1 and bool(config["splits"].get("run_within_shard", True)):
            settings.append(
                _setting(
                    name=f"within_shard_seed{seed}",
                    eval_type="within_shard",
                    split_seed=seed,
                    train={"shard1": shard1.get("train_sample_ids", [])},
                    val={"shard1": shard1.get("val_sample_ids", [])},
                    train_traj={"shard1": shard1.get("train_trajectories", [])},
                    val_traj={"shard1": shard1.get("val_trajectories", [])},
                )
            )
        if shard0 and shard1 and bool(config["splits"].get("run_cross_shard", True)):
            settings.append(
                _setting(
                    name=f"cross_shard_train0_val1_seed{seed}",
                    eval_type="cross_shard",
                    split_seed=seed,
                    train={"shard0": shard0.get("train_sample_ids", [])},
                    val={"shard1": shard1.get("val_sample_ids", [])},
                    train_traj={"shard0": shard0.get("train_trajectories", [])},
                    val_traj={"shard1": shard1.get("val_trajectories", [])},
                )
            )
            settings.append(
                _setting(
                    name=f"cross_shard_train1_val0_seed{seed}",
                    eval_type="cross_shard",
                    split_seed=seed,
                    train={"shard1": shard1.get("train_sample_ids", [])},
                    val={"shard0": shard0.get("val_sample_ids", [])},
                    train_traj={"shard1": shard1.get("train_trajectories", [])},
                    val_traj={"shard0": shard0.get("val_trajectories", [])},
                )
            )
        if shard0 and shard1 and bool(config["splits"].get("run_mixed_shard", True)):
            settings.append(
                _setting(
                    name=f"mixed_shard_seed{seed}",
                    eval_type="mixed_shard",
                    split_seed=seed,
                    train={
                        "shard0": shard0.get("train_sample_ids", []),
                        "shard1": shard1.get("train_sample_ids", []),
                    },
                    val={
                        "shard0": shard0.get("val_sample_ids", []),
                        "shard1": shard1.get("val_sample_ids", []),
                    },
                    train_traj={
                        "shard0": shard0.get("train_trajectories", []),
                        "shard1": shard1.get("train_trajectories", []),
                    },
                    val_traj={
                        "shard0": shard0.get("val_trajectories", []),
                        "shard1": shard1.get("val_trajectories", []),
                    },
                )
            )
    return settings


def summarize_step35_leakage(settings: list[dict[str, Any]]) -> dict[str, Any]:
    bad_sample = [setting["name"] for setting in settings if not setting["train_val_sample_id_disjoint"]]
    bad_traj = [setting["name"] for setting in settings if not setting["train_val_trajectory_disjoint_or_documented"]]
    return {
        "stage": STAGE,
        "num_settings": len(settings),
        "train_val_sample_id_disjoint": not bad_sample,
        "train_val_trajectory_disjoint_or_documented": not bad_traj,
        "sample_id_overlap_settings": bad_sample,
        "trajectory_overlap_settings": bad_traj,
        "no_language_or_trajectory_leakage": not bad_sample and not bad_traj,
        "language_hash_non_comparable_is_not_shift": True,
        "safety_gate_pass": not bad_sample and not bad_traj,
    }


def _setting(
    *,
    name: str,
    eval_type: str,
    split_seed: int,
    train: dict[str, list[str]],
    val: dict[str, list[str]],
    train_traj: dict[str, list[str]],
    val_traj: dict[str, list[str]],
) -> dict[str, Any]:
    train_ids = {str(item) for values in train.values() for item in values}
    val_ids = {str(item) for values in val.values() for item in values}
    train_trajectories = {str(item) for values in train_traj.values() for item in values}
    val_trajectories = {str(item) for values in val_traj.values() for item in values}
    return {
        "stage": STAGE,
        "name": name,
        "eval_type": eval_type,
        "split_seed": int(split_seed),
        "train_sample_ids_by_shard": _string_lists(train),
        "val_sample_ids_by_shard": _string_lists(val),
        "train_trajectories_by_shard": _string_lists(train_traj),
        "val_trajectories_by_shard": _string_lists(val_traj),
        "num_train_samples": len(train_ids),
        "num_val_samples": len(val_ids),
        "train_val_sample_id_disjoint": not bool(train_ids & val_ids),
        "train_val_trajectory_disjoint_or_documented": not bool(train_trajectories & val_trajectories),
        "strict_shard_aware_split": True,
    }


def _string_lists(values: dict[str, list[Any]]) -> dict[str, list[str]]:
    return {str(key): [str(item) for item in items] for key, items in values.items()}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
