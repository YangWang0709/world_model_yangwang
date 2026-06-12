"""Extract Step33A true-temporal VideoMAE tokens for Step32 gap0 windows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_clip_cache import clip_cache_to_torch_videos, load_clip_cache_npz, validate_clip_cache
from data.bridgedata_v2_tfds_true_temporal_manifest import (
    select_step32_gap0_windows,
    summarize_gap0_windows,
    write_jsonl,
)
from data.bridgedata_v2_tfds_true_temporal_schema import (
    TOKENIZATION_MODE,
    pack_temporal_clip,
    summarize_tokens,
    summarize_true_temporal_manifest,
    write_true_temporal_manifest,
)
from encoders.frozen_video_encoder import build_frozen_video_encoder

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_true_temporal_step33a.yaml"


def extract_step33a_true_temporal_tokens(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(config_path)
    env_guard = _env_guard()
    output = config["output"]
    token_summary_path = Path(output["true_temporal_token_summary_json"])
    token_summary_md = Path(output["true_temporal_token_summary_md"])

    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        summary = _safe_stop_summary(config, "env_isaaclab is polluted with TensorFlow/TFDS", env_guard)
        return _write_summary(summary, token_summary_path, token_summary_md)
    missing = _missing_required_inputs(config)
    if missing:
        summary = _safe_stop_summary(config, f"missing Step33A token extraction inputs: {missing}", env_guard)
        return _write_summary(summary, token_summary_path, token_summary_md)

    windows = select_step32_gap0_windows(
        config["input"]["step32_horizon_window_manifest_jsonl"],
        max_windows=int(config["selection"].get("max_windows", 64)),
    )
    if not windows:
        summary = _safe_stop_summary(config, "Step32 gap0 windows are empty", env_guard)
        return _write_summary(summary, token_summary_path, token_summary_md)
    write_jsonl(windows, output["gap0_window_manifest_jsonl"])
    _write_json(Path(output["gap0_window_summary_json"]), summarize_gap0_windows(windows))

    clip_summary = _resolve_gap0_clip_summary(config, windows)
    if bool(clip_summary.get("safe_stop")) or not bool(clip_summary.get("clip_export_performed")):
        summary = _safe_stop_summary(config, clip_summary.get("reason") or "gap0 clip cache unavailable", env_guard)
        summary["clip_export_summary"] = clip_summary
        return _write_summary(summary, token_summary_path, token_summary_md)

    model_path = Path(config["input"]["local_videomae_model"])
    if not (model_path.exists() and (model_path / "config.json").exists()):
        summary = _safe_stop_summary(config, f"local VideoMAE model missing: {model_path}", env_guard)
        summary["model_missing"] = True
        summary["clip_export_summary"] = clip_summary
        return _write_summary(summary, token_summary_path, token_summary_md)

    encoder = build_frozen_video_encoder(_encoder_config(config, str(model_path)))
    if not encoder.is_available():
        summary = _safe_stop_summary(config, f"local VideoMAE unavailable: {encoder.availability.get('reason')}", env_guard)
        summary.update(
            {
                "model_missing": True,
                "encoder_availability": encoder.availability,
                "clip_export_summary": clip_summary,
            }
        )
        return _write_summary(summary, token_summary_path, token_summary_md)

    token_dir = Path(output["true_temporal_token_dir"])
    token_dir.mkdir(parents=True, exist_ok=True)
    for old in token_dir.glob("*.pt"):
        old.unlink()
    manifest_path = Path(output["true_temporal_token_manifest_jsonl"])
    if manifest_path.exists():
        manifest_path.unlink()

    selected_ids = [str(window["sample_id"]) for window in windows]
    clip_by_id = _clip_cache_by_sample_id(clip_summary.get("clip_cache_files", []))
    records: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    required_frames = int(config["true_temporal_token_extraction"].get("video_mae_expected_input_frames", 16))
    image_size = int(config["true_temporal_token_extraction"].get("image_size", 224))

    with torch.no_grad():
        for sample_id in selected_ids:
            clip_path = clip_by_id.get(sample_id)
            if clip_path is None:
                raise FileNotFoundError(f"missing gap0 clip cache for sample_id={sample_id}")
            sample = load_clip_cache_npz(clip_path)
            if not validate_clip_cache(sample, expected_context=16, expected_current=4, expected_future=4):
                raise ValueError(f"invalid gap0 clip cache for Step33A: {clip_path}")
            videos = clip_cache_to_torch_videos(sample, image_size=image_size)
            metadata = dict(videos["metadata"])
            context_tokens, context_raw_shape = _encode_true_temporal(encoder, videos["context_video"], required_frames)
            current_tokens, current_raw_shape = _encode_true_temporal(encoder, videos["current_video"], required_frames)
            future_tokens, future_raw_shape = _encode_true_temporal(encoder, videos["future_video"], required_frames)
            context_summary = summarize_tokens(context_tokens)
            current_summary = summarize_tokens(current_tokens)
            future_summary = summarize_tokens(future_tokens)
            artifact_path = token_dir / f"{sample_id}.pt"
            raw_token_shapes = {
                "context": context_raw_shape,
                "current": current_raw_shape,
                "future": future_raw_shape,
            }
            torch.save(
                {
                    "schema_version": "0.1.0",
                    "stage": config["stage"],
                    "sample_id": sample_id,
                    "trajectory_id": metadata.get("trajectory_id"),
                    "horizon_gap": 0,
                    "tokenization_mode": TOKENIZATION_MODE,
                    "context_tokens": context_tokens.to(dtype=torch.float16).cpu(),
                    "current_tokens": current_tokens.to(dtype=torch.float16).cpu(),
                    "future_tokens": future_tokens.to(dtype=torch.float16).cpu(),
                    "context_summary": context_summary.to(dtype=torch.float16).cpu(),
                    "current_summary": current_summary.to(dtype=torch.float16).cpu(),
                    "future_summary": future_summary.to(dtype=torch.float16).cpu(),
                    "future_delta_last_minus_current": (future_summary - current_summary).to(dtype=torch.float16).cpu(),
                    "future_delta_mean_minus_current": (future_summary - current_summary).to(dtype=torch.float16).cpu(),
                    "raw_token_shapes": raw_token_shapes,
                    "metadata": {
                        **metadata,
                        "tokenization_mode": TOKENIZATION_MODE,
                        "required_input_frames": required_frames,
                        "current_future_padding_mode": "deterministic_repeat_to_required_length",
                        "preserve_frame_order": True,
                        "action_used_as_input": False,
                        "language_used_as_input": False,
                        "goal_used_as_input": False,
                    },
                    "encoder_availability": encoder.availability,
                },
                artifact_path,
            )
            temporal_info = _temporal_bin_info(list(context_tokens.shape))
            records.append(
                {
                    "sample_id": sample_id,
                    "trajectory_id": metadata.get("trajectory_id"),
                    "horizon_gap": 0,
                    "tokenization_mode": TOKENIZATION_MODE,
                    "context_token_shape": list(context_tokens.shape),
                    "current_token_shape": list(current_tokens.shape),
                    "future_token_shape": list(future_tokens.shape),
                    "raw_token_shapes": raw_token_shapes,
                    "token_artifact_path": str(artifact_path),
                    "video_mae_local_only": True,
                    "model_download_performed": False,
                    "temporal_bins_available": temporal_info["temporal_bins_available"],
                    "num_temporal_bins": temporal_info.get("num_temporal_bins"),
                    "num_spatial_tokens_per_bin": temporal_info.get("num_spatial_tokens_per_bin"),
                    "temporal_bin_reason": temporal_info.get("reason"),
                    "image_field": metadata.get("image_field"),
                    "action_used_as_input": False,
                    "language_used_as_input": False,
                    "goal_used_as_input": False,
                }
            )
            model_summaries.append(dict(getattr(encoder.impl, "last_encode_summary", {})))

    write_true_temporal_manifest(records, manifest_path)
    manifest_summary = summarize_true_temporal_manifest(records)
    first_temporal = records[0] if records else {}
    summary = {
        "stage": config["stage"],
        "token_extraction_performed": bool(records),
        "limited_true_temporal_token_extraction_performed": bool(records),
        "safe_stop": not bool(records),
        "reason": None if records else "No Step33A true temporal token records were generated.",
        "num_samples": len(records),
        "horizon_gap": 0,
        "tokenization_mode": TOKENIZATION_MODE,
        "model_loaded_local_only": True,
        "model_path": str(model_path),
        "model_download_performed": False,
        "videomae_training_performed": False,
        "training_performed": False,
        "raw_token_shapes": {
            "context_example": manifest_summary["raw_token_shapes"]["context"] if records else None,
            "current_example": manifest_summary["raw_token_shapes"]["current"] if records else None,
            "future_example": manifest_summary["raw_token_shapes"]["future"] if records else None,
        },
        "summary_shapes": manifest_summary["summary_shapes"],
        "context_token_shape_example": manifest_summary["context_token_shape_example"],
        "current_token_shape_example": manifest_summary["current_token_shape_example"],
        "future_token_shape_example": manifest_summary["future_token_shape_example"],
        "temporal_bins_available": bool(first_temporal.get("temporal_bins_available", False)),
        "num_temporal_bins": first_temporal.get("num_temporal_bins"),
        "num_spatial_tokens_per_bin": first_temporal.get("num_spatial_tokens_per_bin"),
        "data_token_shards_written": False,
        "large_token_shards_generated": False,
        "save_raw_images": False,
        "save_video": False,
        "clip_cache_reused_from_step32": bool(clip_summary.get("clip_cache_reused_from_step32")),
        "limited_clip_export_performed": bool(clip_summary.get("limited_clip_export_performed", False)),
        "clip_export_summary": clip_summary,
        "token_manifest_jsonl": str(manifest_path),
        "true_temporal_token_dir": str(token_dir),
        "encoder_availability": encoder.availability,
        "last_encode_summary": model_summaries[-1] if model_summaries else {},
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": True,
    }
    return _write_summary(summary, token_summary_path, token_summary_md)


def _resolve_gap0_clip_summary(config: dict[str, Any], windows: list[dict[str, Any]]) -> dict[str, Any]:
    output = config["output"]
    selected_ids = {str(window["sample_id"]) for window in windows}
    step32_summary_path = Path(config["input"]["step32_clip_export_summary_json"])
    if bool(config["true_temporal_token_extraction"].get("use_existing_clip_cache_if_available", True)):
        step32_summary = _read_json(step32_summary_path) if step32_summary_path.exists() else {}
        clip_by_id = _clip_cache_by_sample_id(step32_summary.get("clip_cache_files", []))
        if selected_ids and selected_ids.issubset(set(clip_by_id)):
            files = [clip_by_id[str(window["sample_id"])] for window in windows]
            summary = {
                "stage": "bridgedata_v2_tfds_true_temporal_gap0_clip_cache",
                "clip_export_performed": True,
                "limited_clip_export_performed": False,
                "clip_cache_reused_from_step32": True,
                "safe_stop": False,
                "reason": None,
                "num_windows_requested": len(windows),
                "num_windows_exported": len(files),
                "clip_cache_dir": step32_summary.get("clip_cache_dir"),
                "clip_cache_files": files,
                "clip_cache_total_bytes": sum(Path(path).stat().st_size for path in files if Path(path).exists()),
                "download_performed": False,
                "new_tfds_shard_downloaded": False,
                "model_download_performed": False,
                "training_performed": False,
                "token_extraction_performed": False,
                "importance_generation_performed": False,
                "safety_gate_pass": True,
            }
            _write_json(Path(output["clip_export_summary_json"]), summary)
            return summary
    if not bool(config["true_temporal_token_extraction"].get("allow_limited_clip_export_from_existing_shard_if_missing")):
        return {
            "stage": "bridgedata_v2_tfds_true_temporal_gap0_clip_cache",
            "clip_export_performed": False,
            "safe_stop": True,
            "reason": "Step32 gap0 clip cache missing and limited re-export is disabled",
            "clip_cache_files": [],
            "safety_gate_pass": True,
        }
    return _run_limited_gap0_clip_export(config, windows)


def _run_limited_gap0_clip_export(config: dict[str, Any], windows: list[dict[str, Any]]) -> dict[str, Any]:
    output = config["output"]
    write_jsonl(windows, output["gap0_window_manifest_jsonl"])
    adapter = {
        "stage": config["stage"],
        "input": {
            "tfds_dataset_root": config["input"]["tfds_dataset_root"],
            "resolved_fields_json": config["input"]["resolved_fields_json"],
            "resolved_window_manifest_jsonl": output["gap0_window_manifest_jsonl"],
        },
        "dry_run_limits": {
            "max_windows": len(windows),
            "max_samples": len(windows),
            "batch_size": 1,
        },
        "clip_export": {
            "image_size": int(config["true_temporal_token_extraction"].get("image_size", 224)),
            "use_action_as_input": False,
            "use_language_as_input": False,
            "use_goal_image_as_input": False,
        },
        "output": {
            "clip_cache_dir": output["clip_cache_dir"],
            "clip_export_summary_json": output["clip_export_summary_json"],
        },
    }
    adapter_path = Path(output["adapter_config_yaml"])
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text(yaml.safe_dump(adapter, sort_keys=False), encoding="utf-8")
    tfds_python = Path(config["input"]["tfds_env_python"])
    if not tfds_python.exists():
        return {
            "stage": "bridgedata_v2_tfds_true_temporal_gap0_clip_cache",
            "clip_export_performed": False,
            "safe_stop": True,
            "reason": f"TFDS env python missing: {tfds_python}",
            "clip_cache_files": [],
            "safety_gate_pass": True,
        }
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
        raise RuntimeError(f"Step33A limited gap0 clip export failed with code {result.returncode}")
    summary = _read_json(output["clip_export_summary_json"])
    summary.update(
        {
            "stage": "bridgedata_v2_tfds_true_temporal_gap0_clip_cache",
            "limited_clip_export_performed": bool(summary.get("clip_export_performed")),
            "clip_cache_reused_from_step32": False,
            "new_tfds_shard_downloaded": False,
        }
    )
    _write_json(Path(output["clip_export_summary_json"]), summary)
    return summary


def _encode_true_temporal(encoder: Any, video: torch.Tensor, required_frames: int) -> tuple[torch.Tensor, list[int]]:
    packed = pack_temporal_clip(video, required_frames=required_frames)
    encoded = encoder.encode(packed.unsqueeze(0))
    raw_shape = list(encoded.shape)
    tokens = encoded.squeeze(0).detach().to(dtype=torch.float32, device="cpu").contiguous()
    if tokens.ndim != 2:
        raise ValueError(f"true temporal VideoMAE tokens must be [N,D], got {list(tokens.shape)}")
    return tokens, raw_shape


def _clip_cache_by_sample_id(paths: list[str]) -> dict[str, str]:
    by_id: dict[str, str] = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            sample = load_clip_cache_npz(p)
        except Exception:
            continue
        sample_id = str((sample.get("metadata") or {}).get("sample_id") or p.stem)
        by_id[sample_id] = str(p)
    return by_id


def _encoder_config(config: dict[str, Any], model_path: str) -> dict[str, Any]:
    token_cfg = config["true_temporal_token_extraction"]
    return {
        "name": "videomae",
        "model_name_or_path": model_path,
        "allow_download": False,
        "local_files_only": True,
        "image_size": int(token_cfg.get("image_size", 224)),
        "num_frames": int(token_cfg.get("video_mae_expected_input_frames", 16)),
        "output_mode": "last_hidden_state",
        "device": str(token_cfg.get("device", "cuda_if_available")),
        "ignore_mismatched_sizes": True,
    }


def _temporal_bin_info(shape: list[int]) -> dict[str, Any]:
    if len(shape) == 2 and int(shape[0]) > 0 and int(shape[0]) % 196 == 0:
        return {
            "temporal_bins_available": True,
            "num_temporal_bins": int(shape[0]) // 196,
            "num_spatial_tokens_per_bin": 196,
        }
    return {
        "temporal_bins_available": False,
        "reason": "model output shape is treated as flat clip tokens",
    }


def _missing_required_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["input"]["tfds_dataset_root"],
        config["input"]["tfds_env_python"],
        config["input"]["resolved_fields_json"],
        config["input"]["local_videomae_model"],
        config["input"]["step32_horizon_window_manifest_jsonl"],
        config["input"]["step32_horizon_splits_json"],
        config["input"]["step32_horizon_target_summary_json"],
        config["input"]["step32_decision_json"],
    ]
    return [str(path) for path in required if not Path(path).exists()]


def _safe_stop_summary(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    return {
        "stage": config["stage"],
        "token_extraction_performed": False,
        "limited_true_temporal_token_extraction_performed": False,
        "safe_stop": True,
        "reason": reason,
        "num_samples": 0,
        "horizon_gap": 0,
        "tokenization_mode": TOKENIZATION_MODE,
        "model_loaded_local_only": False,
        "model_download_performed": False,
        "videomae_training_performed": False,
        "training_performed": False,
        "raw_token_shapes": {"context_example": None, "current_example": None, "future_example": None},
        "summary_shapes": {"context_summary": [768], "current_summary": [768], "future_summary": [768]},
        "data_token_shards_written": False,
        "large_token_shards_generated": False,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }


def _write_summary(summary: dict[str, Any], json_path: Path, md_path: Path) -> dict[str, Any]:
    _write_json(json_path, summary)
    lines = [
        "# BridgeData V2 Step33A True-Temporal Token Summary",
        "",
        f"- token_extraction_performed: `{str(summary.get('token_extraction_performed', False)).lower()}`",
        f"- safe_stop: `{str(summary.get('safe_stop', False)).lower()}`",
        f"- num_samples: `{summary.get('num_samples')}`",
        f"- tokenization_mode: `{summary.get('tokenization_mode')}`",
        f"- context_token_shape_example: `{summary.get('context_token_shape_example')}`",
        f"- current_token_shape_example: `{summary.get('current_token_shape_example')}`",
        f"- future_token_shape_example: `{summary.get('future_token_shape_example')}`",
        f"- raw_token_shapes: `{summary.get('raw_token_shapes')}`",
        f"- summary_shapes: `{summary.get('summary_shapes')}`",
        f"- model_loaded_local_only: `{str(summary.get('model_loaded_local_only', False)).lower()}`",
        f"- model_download_performed: `{str(summary.get('model_download_performed', False)).lower()}`",
        f"- videomae_training_performed: `{str(summary.get('videomae_training_performed', False)).lower()}`",
        f"- data_token_shards_written: `{str(summary.get('data_token_shards_written', False)).lower()}`",
    ]
    if summary.get("reason"):
        lines.append(f"- reason: `{summary['reason']}`")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _env_guard() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


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
    print(json.dumps(extract_step33a_true_temporal_tokens(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
