"""Generate Step25 BridgeData TFDS context-token importance smoke labels."""

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

from data.bridgedata_v2_tfds_importance_manifest import (
    summarize_importance_manifest,
    write_importance_manifest_jsonl,
)
from data.bridgedata_v2_tfds_importance_proxy import METHOD, compute_context_predictive_importance
from data.bridgedata_v2_tfds_token_artifact_loader import iter_step24_token_samples

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_importance_step25.yaml"


def generate_bridgedata_v2_tfds_importance_from_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(config_path)
    output = config["output"]
    summary_json = Path(output["importance_summary_json"])
    summary_md = Path(output["importance_summary_md"])
    manifest_path = Path(output["importance_manifest_jsonl"])
    importance_dir = Path(output["importance_dir"])

    env_guard = _env_isaaclab_guard()
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        summary = _safe_stop_summary(config, "env_isaaclab is polluted with TensorFlow/TFDS", env_guard)
        return _write_summary(summary, summary_json, summary_md)

    missing = _missing_step24_inputs(config)
    if missing:
        summary = _safe_stop_summary(config, f"missing Step24 token artifacts: {missing}", env_guard)
        return _write_summary(summary, summary_json, summary_md)

    max_samples = int(config["dry_run_limits"].get("max_samples", 4))
    if bool(config["dry_run_limits"].get("save_importance_artifacts", True)):
        importance_dir.mkdir(parents=True, exist_ok=True)
        for old in importance_dir.glob("*.pt"):
            old.unlink()
    if manifest_path.exists():
        manifest_path.unlink()

    records: list[dict[str, Any]] = []
    raw_means: list[float] = []
    raw_stds: list[float] = []
    norm_mins: list[float] = []
    norm_maxs: list[float] = []
    max_artifact_mb = float(config["dry_run_limits"].get("max_importance_artifact_mb", 256))
    token_manifest = Path(config["input"]["token_manifest_jsonl"])

    for sample in iter_step24_token_samples(token_manifest, max_samples=max_samples):
        result = compute_context_predictive_importance(
            sample["context_tokens"],
            sample["current_tokens"],
            sample["future_tokens"],
            context_weight=float(config["proxy_teacher"]["base_prediction"].get("context_weight", 0.25)),
            current_weight=float(config["proxy_teacher"]["base_prediction"].get("current_weight", 0.75)),
        )
        sample_id = str(sample["sample_id"])
        artifact_path = importance_dir / f"{sample_id}_importance.pt"
        metadata = {
            "token_artifact_path": sample["token_artifact_path"],
            "action_used_as_input": False,
            "language_used_as_input": False,
            "goal_used_as_input": False,
            "current_tokens_kept_full": True,
            "train_current_importance": False,
            "label_quality_note": "proxy dry-run only; not final teacher label",
        }
        torch.save(
            {
                "schema_version": "0.1.0",
                "stage": config["stage"],
                "sample_id": sample_id,
                "trajectory_id": sample.get("trajectory_id"),
                "method": METHOD,
                "context_importance_raw": result["context_importance_raw"],
                "context_importance_norm": result["context_importance_norm"],
                "temporal_importance": result["temporal_importance"],
                "spatial_importance": result["spatial_importance"],
                "stats": result["stats"],
                "metadata": metadata,
            },
            artifact_path,
        )
        artifact_mb = artifact_path.stat().st_size / (1024 * 1024)
        if artifact_mb > max_artifact_mb:
            raise RuntimeError(f"Step25 importance artifact exceeds limit: {artifact_path} ({artifact_mb:.2f} MB)")
        stats = result["stats"]
        raw_means.append(float(stats["importance_raw_mean"]))
        raw_stds.append(float(stats["importance_raw_std"]))
        norm_mins.append(float(stats["importance_norm_min"]))
        norm_maxs.append(float(stats["importance_norm_max"]))
        records.append(
            {
                "sample_id": sample_id,
                "trajectory_id": sample.get("trajectory_id"),
                "importance_artifact_path": str(artifact_path),
                "method": METHOD,
                "context_importance_shape": list(result["context_importance_norm"].shape),
                "temporal_importance_shape": list(result["temporal_importance"].shape),
                "spatial_importance_shape": list(result["spatial_importance"].shape),
                "current_tokens_kept_full": True,
                "train_current_importance": False,
                "action_used_as_input": False,
                "language_used_as_input": False,
                "goal_used_as_input": False,
            }
        )

    write_importance_manifest_jsonl(records, manifest_path)
    manifest_summary = summarize_importance_manifest(records)
    summary = {
        "stage": config["stage"],
        "importance_generation_performed": bool(records),
        "safe_stop": not bool(records),
        "reason": None if records else "No Step25 importance records were generated.",
        "num_samples": manifest_summary["num_samples"],
        "num_importance_artifacts": manifest_summary["num_importance_artifacts"],
        "method": METHOD,
        "context_importance_shape_example": manifest_summary["context_importance_shape_example"],
        "temporal_importance_shape_example": manifest_summary["temporal_importance_shape_example"],
        "spatial_importance_shape_example": manifest_summary["spatial_importance_shape_example"],
        "importance_raw_mean": _mean(raw_means),
        "importance_raw_std": _mean(raw_stds),
        "importance_norm_min": min(norm_mins) if norm_mins else None,
        "importance_norm_max": max(norm_maxs) if norm_maxs else None,
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "download_performed": False,
        "model_download_performed": False,
        "training_performed": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "world_model_training_performed": False,
        "token_extraction_performed": False,
        "large_importance_shards_generated": False,
        "data_importance_shards_written": False,
        "data_token_shards_written": False,
        "current_importance_generated": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
        "label_quality_note": "proxy dry-run only; not final teacher label",
        "importance_manifest_jsonl": str(manifest_path),
        "importance_dir": str(importance_dir),
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": True,
    }
    return _write_summary(summary, summary_json, summary_md)


def _missing_step24_inputs(config: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in ("token_manifest_jsonl", "token_summary_json", "token_smoke_dir"):
        path = Path(config["input"][key])
        if not path.exists():
            missing.append(str(path))
    if not missing:
        token_summary = json.loads(Path(config["input"]["token_summary_json"]).read_text(encoding="utf-8"))
        if not bool(token_summary.get("token_extraction_performed")):
            missing.append("Step24 token_summary_json reports token_extraction_performed=false")
    return missing


def _safe_stop_summary(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    return {
        "stage": config["stage"],
        "importance_generation_performed": False,
        "safe_stop": True,
        "reason": reason,
        "num_samples": 0,
        "num_importance_artifacts": 0,
        "method": METHOD,
        "context_importance_shape_example": None,
        "temporal_importance_shape_example": None,
        "spatial_importance_shape_example": None,
        "importance_raw_mean": None,
        "importance_raw_std": None,
        "importance_norm_min": None,
        "importance_norm_max": None,
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "download_performed": False,
        "model_download_performed": False,
        "training_performed": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "world_model_training_performed": False,
        "token_extraction_performed": False,
        "large_importance_shards_generated": False,
        "data_importance_shards_written": False,
        "data_token_shards_written": False,
        "current_importance_generated": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
        "label_quality_note": "proxy dry-run only; not final teacher label",
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }


def _write_summary(summary: dict[str, Any], json_path: Path, md_path: Path) -> dict[str, Any]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# BridgeData V2 TFDS Importance Summary",
        "",
        f"- importance_generation_performed: `{str(summary.get('importance_generation_performed', False)).lower()}`",
        f"- safe_stop: `{str(summary.get('safe_stop', False)).lower()}`",
        f"- method: `{summary.get('method')}`",
        f"- num_samples: `{summary.get('num_samples')}`",
        f"- context_importance_shape_example: `{summary.get('context_importance_shape_example')}`",
        f"- temporal_importance_shape_example: `{summary.get('temporal_importance_shape_example')}`",
        f"- spatial_importance_shape_example: `{summary.get('spatial_importance_shape_example')}`",
        f"- importance_norm_min: `{summary.get('importance_norm_min')}`",
        f"- importance_norm_max: `{summary.get('importance_norm_max')}`",
        f"- current_tokens_kept_full: `{str(summary.get('current_tokens_kept_full', False)).lower()}`",
        f"- train_current_importance: `{str(summary.get('train_current_importance', True)).lower()}`",
        f"- token_extraction_performed: `{str(summary.get('token_extraction_performed', False)).lower()}`",
        f"- training_performed: `{str(summary.get('training_performed', False)).lower()}`",
        f"- label_quality_note: `{summary.get('label_quality_note')}`",
    ]
    if summary.get("reason"):
        lines.append(f"- reason: `{summary['reason']}`")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _env_isaaclab_guard() -> dict[str, bool]:
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
    print(json.dumps(generate_bridgedata_v2_tfds_importance_from_config(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
