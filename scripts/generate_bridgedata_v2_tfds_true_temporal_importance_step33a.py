"""Generate Step33A proxy importance for true-temporal context clip tokens."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_true_temporal_proxy_importance import (
    METHOD,
    compute_true_temporal_context_importance,
    summarize_true_temporal_importance,
    write_true_temporal_importance_manifest,
)
from data.bridgedata_v2_tfds_true_temporal_schema import (
    load_true_temporal_token_artifact,
    read_true_temporal_manifest,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_true_temporal_step33a.yaml"


def generate_step33a_true_temporal_importance(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(config_path)
    env_guard = _env_guard()
    output = config["output"]
    summary_json = Path(output["true_temporal_importance_summary_json"])
    summary_md = Path(output["true_temporal_importance_summary_md"])

    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        summary = _safe_stop_summary(config, "env_isaaclab is polluted with TensorFlow/TFDS", env_guard)
        return _write_summary(summary, summary_json, summary_md)
    missing = _missing_inputs(config)
    if missing:
        summary = _safe_stop_summary(config, f"missing Step33A importance inputs: {missing}", env_guard)
        return _write_summary(summary, summary_json, summary_md)

    token_records = read_true_temporal_manifest(output["true_temporal_token_manifest_jsonl"])
    importance_dir = Path(output["true_temporal_importance_dir"])
    importance_dir.mkdir(parents=True, exist_ok=True)
    for old in importance_dir.glob("*.pt"):
        old.unlink()
    manifest_path = Path(output["true_temporal_importance_manifest_jsonl"])
    if manifest_path.exists():
        manifest_path.unlink()

    records: list[dict[str, Any]] = []
    raw_means: list[float] = []
    raw_stds: list[float] = []
    norm_mins: list[float] = []
    norm_maxs: list[float] = []
    for record in token_records:
        sample = load_true_temporal_token_artifact(record)
        result = compute_true_temporal_context_importance(
            sample["context_tokens"],
            sample["current_summary"],
            sample["future_summary"],
        )
        sample_id = str(sample["sample_id"])
        artifact_path = importance_dir / f"{sample_id}_importance.pt"
        metadata = {
            "token_artifact_path": sample["token_artifact_path"],
            "tokenization_mode": sample["tokenization_mode"],
            "current_tokens_kept_full": True,
            "train_current_importance": False,
            "current_importance_generated": False,
            "action_used_as_input": False,
            "language_used_as_input": False,
            "goal_used_as_input": False,
            "label_quality_note": "proxy dry-run only; not final teacher label",
        }
        torch.save(
            {
                "schema_version": "0.1.0",
                "stage": config["stage"],
                "sample_id": sample_id,
                "trajectory_id": sample.get("trajectory_id"),
                "horizon_gap": 0,
                "method": METHOD,
                "context_importance_raw": result["context_importance_raw"],
                "context_importance_norm": result["context_importance_norm"],
                "temporal_importance": result["temporal_importance"],
                "spatial_importance": result["spatial_importance"],
                "temporal_spatial_summary_available": result["temporal_spatial_summary_available"],
                "stats": result["stats"],
                "metadata": metadata,
            },
            artifact_path,
        )
        stats = result["stats"]
        raw_means.append(float(stats["importance_raw_mean"]))
        raw_stds.append(float(stats["importance_raw_std"]))
        norm_mins.append(float(stats["importance_norm_min"]))
        norm_maxs.append(float(stats["importance_norm_max"]))
        records.append(
            {
                "sample_id": sample_id,
                "trajectory_id": sample.get("trajectory_id"),
                "horizon_gap": 0,
                "importance_artifact_path": str(artifact_path),
                "method": METHOD,
                "context_importance_shape": list(result["context_importance_norm"].shape),
                "temporal_importance_shape": list(result["temporal_importance"].shape)
                if result["temporal_importance"] is not None
                else None,
                "spatial_importance_shape": list(result["spatial_importance"].shape)
                if result["spatial_importance"] is not None
                else None,
                "temporal_spatial_summary_available": bool(result["temporal_spatial_summary_available"]),
                "current_tokens_kept_full": True,
                "train_current_importance": False,
                "current_importance_generated": False,
                "selector_training_performed": False,
                "action_used_as_input": False,
                "language_used_as_input": False,
                "goal_used_as_input": False,
            }
        )

    write_true_temporal_importance_manifest(records, manifest_path)
    manifest_summary = summarize_true_temporal_importance(records)
    summary = {
        "stage": config["stage"],
        "importance_generation_performed": bool(records),
        "limited_true_temporal_proxy_importance_performed": bool(records),
        "safe_stop": not bool(records),
        "reason": None if records else "No Step33A true-temporal importance records were generated.",
        "num_samples": manifest_summary["num_samples"],
        "method": METHOD,
        "context_importance_shape_example": manifest_summary["context_importance_shape_example"],
        "temporal_importance_shape_example": manifest_summary["temporal_importance_shape_example"],
        "spatial_importance_shape_example": manifest_summary["spatial_importance_shape_example"],
        "temporal_spatial_summary_available": manifest_summary["temporal_spatial_summary_available"],
        "importance_raw_mean": _mean(raw_means),
        "importance_raw_std": _mean(raw_stds),
        "importance_norm_min": min(norm_mins) if norm_mins else None,
        "importance_norm_max": max(norm_maxs) if norm_maxs else None,
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "current_importance_generated": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "videomae_training_performed": False,
        "training_performed": False,
        "model_download_performed": False,
        "data_importance_shards_written": False,
        "data_token_shards_written": False,
        "large_importance_shards_generated": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
        "label_quality_note": "proxy dry-run only; not final teacher label",
        "importance_manifest_jsonl": str(manifest_path),
        "true_temporal_importance_dir": str(importance_dir),
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": True,
    }
    return _write_summary(summary, summary_json, summary_md)


def _missing_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["output"]["true_temporal_token_manifest_jsonl"],
        config["output"]["true_temporal_token_summary_json"],
        config["output"]["true_temporal_token_dir"],
    ]
    return [str(path) for path in required if not Path(path).exists()]


def _safe_stop_summary(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    return {
        "stage": config["stage"],
        "importance_generation_performed": False,
        "limited_true_temporal_proxy_importance_performed": False,
        "safe_stop": True,
        "reason": reason,
        "num_samples": 0,
        "method": METHOD,
        "context_importance_shape_example": None,
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "current_importance_generated": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "training_performed": False,
        "model_download_performed": False,
        "data_importance_shards_written": False,
        "data_token_shards_written": False,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }


def _write_summary(summary: dict[str, Any], json_path: Path, md_path: Path) -> dict[str, Any]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# BridgeData V2 Step33A True-Temporal Importance Summary",
        "",
        f"- importance_generation_performed: `{str(summary.get('importance_generation_performed', False)).lower()}`",
        f"- safe_stop: `{str(summary.get('safe_stop', False)).lower()}`",
        f"- num_samples: `{summary.get('num_samples')}`",
        f"- method: `{summary.get('method')}`",
        f"- context_importance_shape_example: `{summary.get('context_importance_shape_example')}`",
        f"- temporal_importance_shape_example: `{summary.get('temporal_importance_shape_example')}`",
        f"- spatial_importance_shape_example: `{summary.get('spatial_importance_shape_example')}`",
        f"- current_tokens_kept_full: `{str(summary.get('current_tokens_kept_full', False)).lower()}`",
        f"- train_current_importance: `{str(summary.get('train_current_importance', True)).lower()}`",
        f"- selector_training_performed: `{str(summary.get('selector_training_performed', False)).lower()}`",
        f"- data_importance_shards_written: `{str(summary.get('data_importance_shards_written', False)).lower()}`",
    ]
    if summary.get("reason"):
        lines.append(f"- reason: `{summary['reason']}`")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _env_guard() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


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
    print(json.dumps(generate_step33a_true_temporal_importance(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
