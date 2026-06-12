"""Prepare Step29 selected windows and train/val split from existing BridgeData TFDS shard metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_trainval_split import split_selected_windows_from_config
from data.bridgedata_v2_tfds_window_sampler import sample_windows_from_manifest

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_trainval_context_utility_step29.yaml"


def prepare_step29_windows(config_path: str | Path = DEFAULT_CONFIG, use_optional_64: bool = False) -> dict:
    config = _load_yaml(config_path)
    missing = [
        str(path)
        for path in (
            Path(config["input"]["resolved_fields_json"]),
            Path(config["input"]["resolved_window_manifest_jsonl"]),
            Path(config["input"]["tfds_dataset_root"]),
        )
        if not path.exists()
    ]
    if missing:
        payload = _safe_stop_payload(config, f"missing required existing Step23.5/shard inputs: {missing}")
        _write_json(Path(config["output"]["split_json"]), payload)
        return payload

    selection_cfg = config["window_selection"]
    target = (
        int(selection_cfg["target_num_windows_optional"])
        if use_optional_64 and bool(selection_cfg.get("use_optional_64_if_resources_ok", False))
        else int(selection_cfg["target_num_windows_primary"])
    )
    summary = sample_windows_from_manifest(
        config["input"]["resolved_window_manifest_jsonl"],
        config["output"]["selected_windows_jsonl"],
        target_num_windows=target,
        max_windows_hard_cap=int(selection_cfg.get("max_windows_hard_cap", 64)),
    )
    split = split_selected_windows_from_config(config)
    payload = {
        "stage": config["stage"],
        "safe_stop": False,
        "selected_windows_prepared": True,
        "window_selection_summary": summary,
        "split": split,
        "num_selected_windows": summary["num_selected_windows"],
        "num_train_windows": split["num_train_windows"],
        "num_val_windows": split["num_val_windows"],
        "train_val_trajectory_disjoint": split["train_val_trajectory_disjoint"],
        "new_tfds_shard_downloaded": False,
        "download_performed": False,
        "safety_gate_pass": True,
    }
    _write_json(Path(config["output"]["split_json"]), {**split, "window_selection_summary": summary})
    return payload


def _safe_stop_payload(config: dict, reason: str) -> dict:
    return {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "selected_windows_prepared": False,
        "num_selected_windows": 0,
        "num_train_windows": 0,
        "num_val_windows": 0,
        "new_tfds_shard_downloaded": False,
        "download_performed": False,
        "safety_gate_pass": True,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--use-optional-64", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_step29_windows(args.config, use_optional_64=args.use_optional_64), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

