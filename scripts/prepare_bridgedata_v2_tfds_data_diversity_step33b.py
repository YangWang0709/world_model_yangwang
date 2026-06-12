"""Prepare Step33B second-shard windows and splits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_data_diversity_splits import write_step33b_splits
from data.bridgedata_v2_tfds_second_shard_acquisition import (
    acquire_second_shard,
    write_acquisition_summary,
)
from data.bridgedata_v2_tfds_shard_inventory import inventory_local_tfds_shards, write_shard_inventory

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_data_diversity_step33b.yaml"


def prepare_step33b_data_diversity(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    missing = _missing_existing_inputs(config)
    if missing:
        payload = _safe_stop_payload(config, f"missing Step33B prerequisite inputs: {missing}")
        _write_prepare_outputs(config, payload)
        return payload

    shard_cfg = config["second_shard"]
    acquisition = acquire_second_shard(
        dataset_root=config["input"]["tfds_dataset_root"],
        base_url=config["input"]["official_tfds_base_url"],
        preferred_indices=[int(item) for item in shard_cfg.get("preferred_shard_indices", [1])],
        expected_filename_pattern=str(shard_cfg["expected_filename_pattern"]),
        max_new_shards_to_download=int(shard_cfg.get("max_new_shards_to_download", 1)),
        warning_size_mb=int(shard_cfg.get("warning_size_mb", 256)),
        hard_cap_size_mb=int(shard_cfg.get("hard_cap_size_mb", 1024)),
    )
    write_acquisition_summary(acquisition, config["output"]["shard_acquisition_summary_json"])
    inventory = inventory_local_tfds_shards(
        config["input"]["tfds_dataset_root"],
        selected_shard_index=acquisition.get("selected_shard_index"),
    )
    write_shard_inventory(inventory, config["output"]["shard_inventory_json"])
    if bool(acquisition.get("safe_stop")):
        payload = {
            "stage": config["stage"],
            "safe_stop": True,
            "reason": acquisition.get("reason"),
            "shard_acquisition_summary": acquisition,
            "shard_inventory": inventory,
            "shard1_window_summary": _minimal_window_summary(config, acquisition.get("reason")),
            "shard_splits": _minimal_split_summary(config, acquisition.get("reason")),
            "safety_gate_pass": True,
        }
        _write_prepare_outputs(config, payload)
        return payload

    window_summary = _run_tfds_window_builder(config, int(acquisition["selected_shard_index"]))
    if bool(window_summary.get("safe_stop")):
        payload = {
            "stage": config["stage"],
            "safe_stop": True,
            "reason": window_summary.get("reason"),
            "shard_acquisition_summary": acquisition,
            "shard_inventory": inventory,
            "shard1_window_summary": window_summary,
            "shard_splits": _minimal_split_summary(config, window_summary.get("reason")),
            "safety_gate_pass": bool(window_summary.get("safety_gate_pass", True)),
        }
        _write_prepare_outputs(config, payload)
        return payload

    selection_cfg = config["window_selection"]
    splits = write_step33b_splits(
        config["output"]["shard1_window_manifest_jsonl"],
        config["output"]["shard_split_json"],
        seeds=[int(seed) for seed in selection_cfg.get("split_seeds", [42, 123, 999])],
        train_ratio=float(selection_cfg.get("train_ratio", 0.75)),
        val_ratio=float(selection_cfg.get("val_ratio", 0.25)),
        trajectory_disjoint_if_possible=bool(selection_cfg.get("trajectory_disjoint_if_possible", True)),
    )
    return {
        "stage": config["stage"],
        "safe_stop": False,
        "reason": None,
        "shard_acquisition_summary": acquisition,
        "shard_inventory": inventory,
        "shard1_window_summary": window_summary,
        "shard_splits": splits,
        "downloaded_new_shard_count": int(acquisition.get("downloaded_new_shard_count", 0)),
        "safety_gate_pass": True,
    }


def _build_windows_in_current_process(config: dict[str, Any], selected_shard_index: int) -> dict[str, Any]:
    from data.bridgedata_v2_tfds_data_diversity_windows import build_gap0_windows_from_tfds_shard

    window_cfg = config["window"]
    selection_cfg = config["window_selection"]
    return build_gap0_windows_from_tfds_shard(
        dataset_root=config["input"]["tfds_dataset_root"],
        resolved_fields_json=config["input"]["resolved_fields_json"],
        output_manifest_jsonl=config["output"]["shard1_window_manifest_jsonl"],
        output_summary_json=config["output"]["shard1_window_summary_json"],
        shard_index=int(selected_shard_index),
        target_windows=int(selection_cfg.get("target_windows_per_shard", 64)),
        min_windows=int(selection_cfg.get("min_windows_per_shard", 32)),
        max_windows_per_trajectory_soft_cap=int(selection_cfg.get("max_windows_per_trajectory_soft_cap", 16)),
        context_len=int(window_cfg.get("context_len", 16)),
        current_len=int(window_cfg.get("current_len", 4)),
        future_len=int(window_cfg.get("future_len", 4)),
        horizon_gap=int(window_cfg.get("horizon_gap", 0)),
    )


def _run_tfds_window_builder(config: dict[str, Any], selected_shard_index: int) -> dict[str, Any]:
    tfds_python = Path(config["input"]["tfds_env_python"])
    if not tfds_python.exists():
        return _minimal_window_summary(config, f"TFDS env python missing: {tfds_python}")
    command = [
        str(tfds_python),
        str(Path(__file__).resolve()),
        "--config",
        str(DEFAULT_CONFIG),
        "--only-build-windows",
        str(int(selected_shard_index)),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        return _minimal_window_summary(config, f"TFDS window builder failed with code {result.returncode}")
    return json.loads(Path(config["output"]["shard1_window_summary_json"]).read_text(encoding="utf-8"))


def _missing_existing_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["input"]["tfds_dataset_root"],
        config["input"]["resolved_fields_json"],
        config["input"]["local_videomae_model"],
        config["input"]["shard0_path"],
        config["input"]["step32_horizon_window_manifest_jsonl"],
        config["input"]["step32_horizon_splits_json"],
        config["input"]["step32_frame_repeat_token_manifest_jsonl"],
        config["input"]["step32_frame_repeat_token_summary_json"],
        config["input"]["step32_importance_manifest_jsonl"],
        config["input"]["step32_importance_summary_json"],
        config["input"]["step32_horizon_target_summary_json"],
        config["input"]["step32_decision_json"],
        config["input"]["step33a_decision_json"],
    ]
    return [str(path) for path in required if not Path(path).exists()]


def _safe_stop_payload(config: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "shard_acquisition_summary": {
            "stage": config["stage"],
            "safe_stop": True,
            "reason": reason,
            "second_shard_available": False,
            "downloaded_new_shard_count": 0,
            "raw_zip_downloaded": False,
            "full_tfds_downloaded": False,
            "model_download_performed": False,
            "safety_gate_pass": True,
        },
        "shard_inventory": {},
        "shard1_window_summary": _minimal_window_summary(config, reason),
        "shard_splits": _minimal_split_summary(config, reason),
        "safety_gate_pass": True,
    }


def _write_prepare_outputs(config: dict[str, Any], payload: dict[str, Any]) -> None:
    _write_json(Path(config["output"]["shard_acquisition_summary_json"]), payload["shard_acquisition_summary"])
    _write_json(Path(config["output"]["shard_inventory_json"]), payload.get("shard_inventory") or {})
    _write_json(Path(config["output"]["shard1_window_summary_json"]), payload["shard1_window_summary"])
    _write_json(Path(config["output"]["shard_split_json"]), payload["shard_splits"])


def _minimal_window_summary(config: dict[str, Any], reason: str | None) -> dict[str, Any]:
    return {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "selected_windows": 0,
        "num_trajectories": 0,
        "safety_gate_pass": True,
    }


def _minimal_split_summary(config: dict[str, Any], reason: str | None) -> dict[str, Any]:
    return {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "splits": [],
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
    parser.add_argument("--only-build-windows", type=int, default=None)
    args = parser.parse_args()
    config = _load_yaml(args.config)
    if args.only_build_windows is not None:
        print(json.dumps(_build_windows_in_current_process(config, int(args.only_build_windows)), indent=2, sort_keys=True))
    else:
        print(json.dumps(prepare_step33b_data_diversity(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
