"""Prepare Step30A diverse windows and multi-seed splits from existing TFDS metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_diverse_window_sampler import sample_trajectory_diverse_windows_from_manifest
from data.bridgedata_v2_tfds_multiseed_split import write_multiseed_splits_from_manifest

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_window_diversity_step30a.yaml"


def prepare_step30a_window_diversity(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    missing = _missing_existing_inputs(config)
    if missing:
        payload = _safe_stop_payload(config, f"missing existing Step23.5/shard inputs: {missing}")
        _write_json(Path(config["output"]["selection_summary_json"]), payload["selection_summary"])
        _write_json(Path(config["output"]["multiseed_splits_json"]), payload["multiseed_splits"])
        return payload

    diversity = config["window_diversity"]
    target_count = int((diversity.get("selected_counts_to_run") or [diversity["primary_window_count"]])[0])
    selection_summary = sample_trajectory_diverse_windows_from_manifest(
        config["input"]["resolved_window_manifest_jsonl"],
        config["output"]["selected_windows_jsonl"],
        config["output"]["selection_summary_json"],
        target_count=target_count,
        hard_cap_windows=int(diversity.get("hard_cap_windows", 128)),
        max_windows_per_trajectory_soft_cap=int(diversity.get("max_windows_per_trajectory_soft_cap", 16)),
    )
    split_cfg = config["split"]
    multiseed_splits = write_multiseed_splits_from_manifest(
        config["output"]["selected_windows_jsonl"],
        config["output"]["multiseed_splits_json"],
        seeds=[int(seed) for seed in split_cfg.get("seeds", [42, 123, 999])],
        train_ratio=float(split_cfg.get("train_ratio", 0.75)),
        val_ratio=float(split_cfg.get("val_ratio", 0.25)),
        min_val_windows_by_count={int(k): int(v) for k, v in split_cfg.get("min_val_windows_by_count", {}).items()},
    )
    payload = {
        "stage": config["stage"],
        "safe_stop": False,
        "selected_windows_prepared": True,
        "selection_summary": selection_summary,
        "multiseed_splits": multiseed_splits,
        "num_selected_windows": int(selection_summary["num_selected_windows"]),
        "num_selected_trajectories": int(selection_summary["num_selected_trajectories"]),
        "num_splits": int(multiseed_splits["num_splits"]),
        "split_seeds": multiseed_splits["seeds"],
        "new_tfds_shard_downloaded": False,
        "download_performed": False,
        "safety_gate_pass": True,
    }
    return payload


def _missing_existing_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["input"]["resolved_fields_json"],
        config["input"]["resolved_window_manifest_jsonl"],
        config["input"]["tfds_dataset_root"],
    ]
    return [str(path) for path in required if not Path(path).exists()]


def _safe_stop_payload(config: dict[str, Any], reason: str) -> dict[str, Any]:
    selection_summary = {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "num_available_windows": 0,
        "num_selected_windows": 0,
        "num_selected_trajectories": 0,
        "new_tfds_shard_downloaded": False,
        "download_performed": False,
        "safety_gate_pass": True,
    }
    multiseed_splits = {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "num_splits": 0,
        "splits": [],
        "safety_gate_pass": True,
    }
    return {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "selected_windows_prepared": False,
        "selection_summary": selection_summary,
        "multiseed_splits": multiseed_splits,
        "new_tfds_shard_downloaded": False,
        "download_performed": False,
        "safety_gate_pass": True,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(prepare_step30a_window_diversity(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
