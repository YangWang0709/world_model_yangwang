"""Prepare Step32 longer-horizon windows and multi-seed splits."""

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

from data.bridgedata_v2_tfds_longer_horizon_split import write_longer_horizon_splits
from data.bridgedata_v2_tfds_longer_horizon_window_builder import build_longer_horizon_windows_from_manifest

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_longer_horizon_step32.yaml"


def prepare_step32_longer_horizon_windows(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    missing = _missing_existing_inputs(config)
    if missing:
        payload = _safe_stop_payload(config, f"missing existing Step32 inputs: {missing}")
        _write_json(Path(config["output"]["horizon_window_summary_json"]), payload["horizon_window_summary"])
        _write_json(Path(config["output"]["horizon_split_json"]), payload["horizon_splits"])
        return payload

    window_cfg = config["window"]
    selection_cfg = config["window_selection"]
    summary = build_longer_horizon_windows_from_manifest(
        config["input"]["resolved_window_manifest_jsonl"],
        config["output"]["horizon_window_manifest_jsonl"],
        config["output"]["horizon_window_summary_json"],
        horizon_gaps=[int(gap) for gap in window_cfg["horizon_gaps"]],
        required_horizon_gaps=[int(gap) for gap in window_cfg["required_horizon_gaps"]],
        optional_horizon_gaps=[int(gap) for gap in window_cfg.get("optional_horizon_gaps", [])],
        target_windows_per_horizon=int(selection_cfg.get("target_windows_per_horizon", 64)),
        min_windows_per_horizon=int(selection_cfg.get("min_windows_per_horizon", 32)),
        max_windows_per_trajectory_soft_cap=int(selection_cfg.get("max_windows_per_trajectory_soft_cap", 16)),
        deterministic_seed=int(selection_cfg.get("deterministic_seed", 42)),
    )
    splits = write_longer_horizon_splits(
        config["output"]["horizon_window_manifest_jsonl"],
        config["output"]["horizon_split_json"],
        seeds=[int(seed) for seed in selection_cfg.get("split_seeds", [42, 123, 999])],
        train_ratio=float(selection_cfg.get("train_ratio", 0.75)),
        val_ratio=float(selection_cfg.get("val_ratio", 0.25)),
        trajectory_disjoint_if_possible=bool(selection_cfg.get("trajectory_disjoint_if_possible", True)),
    )
    return {
        "stage": config["stage"],
        "safe_stop": bool(summary.get("safe_stop")),
        "reason": summary.get("reason"),
        "longer_horizon_window_builder_performed": True,
        "horizon_window_summary": summary,
        "horizon_splits": splits,
        "new_tfds_shard_downloaded": False,
        "download_performed": False,
        "safety_gate_pass": True,
    }


def _missing_existing_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["input"]["tfds_dataset_root"],
        config["input"]["resolved_fields_json"],
        config["input"]["resolved_window_manifest_jsonl"],
        config["input"]["local_videomae_model"],
        config["input"]["step31_decision_json"],
    ]
    return [str(path) for path in required if not Path(path).exists()]


def _safe_stop_payload(config: dict[str, Any], reason: str) -> dict[str, Any]:
    summary = {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "longer_horizon_window_builder_performed": False,
        "horizons": {},
        "horizons_completed": [],
        "new_tfds_shard_downloaded": False,
        "raw_zip_downloaded": False,
        "download_performed": False,
        "safety_gate_pass": True,
    }
    splits = {
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
        "longer_horizon_window_builder_performed": False,
        "horizon_window_summary": summary,
        "horizon_splits": splits,
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
    print(json.dumps(prepare_step32_longer_horizon_windows(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
