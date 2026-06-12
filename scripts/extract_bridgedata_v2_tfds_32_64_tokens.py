"""Run Step29 limited clip export and local-only VideoMAE token extraction."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_expanded_token_manifest import (
    enrich_token_manifest_with_split,
    summarize_expanded_token_manifest,
)
from scripts.extract_bridgedata_v2_tfds_videomae_tokens import extract_bridgedata_v2_tfds_tokens_from_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_trainval_context_utility_step29.yaml"


def extract_step29_tokens(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(config_path)
    missing = _missing_inputs_for_token_extraction(config)
    if missing:
        summary = _safe_stop_summary(config, f"missing Step29 token extraction inputs: {missing}")
        _write_json(Path(config["output"]["token_summary_json"]), summary)
        return summary

    selected_count = _count_jsonl(config["output"]["selected_windows_jsonl"])
    adapter = _adapter_config(config, selected_count)
    adapter_path = Path(config["output"]["adapter_config_yaml"])
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text(yaml.safe_dump(adapter, sort_keys=False), encoding="utf-8")

    _run_tfds_export(config, adapter_path)
    summary = extract_bridgedata_v2_tfds_tokens_from_config(adapter_path)
    records = []
    if summary.get("token_extraction_performed"):
        records = enrich_token_manifest_with_split(config["output"]["token_manifest_jsonl"], config["output"]["split_json"])
    split_summary = summarize_expanded_token_manifest(records) if records else {}
    summary.update(
        {
            "stage": "bridgedata_v2_tfds_trainval_context_utility_step29_token_extraction",
            "limited_token_extraction_performed": bool(summary.get("token_extraction_performed")),
            "num_windows_tokenized": int(summary.get("num_windows_tokenized") or 0),
            "new_tfds_shard_downloaded": False,
            "model_download_performed": False,
            "token_shapes_ok": _token_shapes_ok(summary),
            "data_token_shards_written": False,
            "split_counts": split_summary.get("split_counts", {}),
            "safety_gate_pass": bool(summary.get("safety_gate_pass", True)),
        }
    )
    _write_json(Path(config["output"]["token_summary_json"]), summary)
    return summary


def _run_tfds_export(config: dict[str, Any], adapter_path: Path) -> None:
    tfds_python = Path(config["input"]["tfds_env_python"])
    if not tfds_python.exists():
        raise RuntimeError(f"TFDS env python missing: {tfds_python}")
    command = [
        str(tfds_python),
        str(PROJECT_ROOT / "scripts" / "export_bridgedata_v2_tfds_resolved_clips.py"),
        "--config",
        str(adapter_path),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"TFDS clip export failed with code {result.returncode}")


def _adapter_config(config: dict[str, Any], selected_count: int) -> dict[str, Any]:
    adapter = dict(config)
    adapter["input"] = dict(config["input"])
    adapter["input"]["resolved_window_manifest_jsonl"] = config["output"]["selected_windows_jsonl"]
    adapter["dry_run_limits"] = dict(config["dry_run_limits"])
    adapter["dry_run_limits"]["max_windows"] = int(selected_count)
    adapter["dry_run_limits"]["max_samples"] = int(selected_count)
    adapter["dry_run_limits"]["batch_size"] = int(config["token_extraction"].get("batch_size", 1))
    adapter["output"] = dict(config["output"])
    adapter["output"]["token_summary_md"] = config["output"]["token_summary_md"]
    return adapter


def _missing_inputs_for_token_extraction(config: dict[str, Any]) -> list[str]:
    required = [
        config["output"]["selected_windows_jsonl"],
        config["output"]["split_json"],
        config["input"]["resolved_fields_json"],
        config["input"]["tfds_dataset_root"],
        config["input"]["local_videomae_model"],
    ]
    return [str(path) for path in required if not Path(path).exists()]


def _safe_stop_summary(config: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "stage": "bridgedata_v2_tfds_trainval_context_utility_step29_token_extraction",
        "limited_token_extraction_performed": False,
        "token_extraction_performed": False,
        "safe_stop": True,
        "reason": reason,
        "num_windows_tokenized": 0,
        "model_loaded_local_only": False,
        "model_download_performed": False,
        "new_tfds_shard_downloaded": False,
        "token_shapes_ok": False,
        "data_token_shards_written": False,
        "safety_gate_pass": True,
    }


def _token_shapes_ok(summary: dict[str, Any]) -> bool:
    return (
        list(summary.get("context_token_shape_example") or []) == [16, 392, 768]
        and list(summary.get("current_token_shape_example") or []) == [4, 392, 768]
        and list(summary.get("future_token_shape_example") or []) == [4, 392, 768]
    )


def _count_jsonl(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


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
    print(json.dumps(extract_step29_tokens(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

