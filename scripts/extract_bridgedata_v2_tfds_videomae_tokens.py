"""Run local-only VideoMAE token extraction on Step24 BridgeData clip caches."""

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

from data.bridgedata_v2_tfds_clip_cache import clip_cache_to_torch_videos, load_clip_cache_npz, validate_clip_cache
from data.bridgedata_v2_tfds_token_manifest import summarize_token_manifest, write_token_manifest_jsonl
from encoders.frozen_video_encoder import build_frozen_video_encoder

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_token_extraction_step24.yaml"


def extract_bridgedata_v2_tfds_tokens_from_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(config_path)
    output_cfg = config["output"]
    clip_summary_path = Path(output_cfg["clip_export_summary_json"])
    token_summary_path = Path(output_cfg["token_summary_json"])
    token_summary_md_path = Path(output_cfg["token_summary_md"])
    token_manifest_path = Path(output_cfg["token_manifest_jsonl"])
    token_smoke_dir = Path(output_cfg["token_smoke_dir"])

    env_guard = _env_isaaclab_guard()
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        summary = _safe_stop_summary(config, "env_isaaclab is polluted with TensorFlow/TFDS", env_guard)
        return _write_token_summary(summary, token_summary_path, token_summary_md_path)

    if not clip_summary_path.exists():
        summary = _safe_stop_summary(config, "clip_export_summary_json is missing", env_guard)
        return _write_token_summary(summary, token_summary_path, token_summary_md_path)
    clip_summary = json.loads(clip_summary_path.read_text(encoding="utf-8"))
    if not clip_summary.get("clip_export_performed"):
        summary = _safe_stop_summary(config, clip_summary.get("reason") or "clip export did not run", env_guard)
        summary["clip_export_summary"] = clip_summary
        return _write_token_summary(summary, token_summary_path, token_summary_md_path)

    model_path = _find_local_model_path(config["videomae"].get("model_name_or_path_candidates", []))
    if model_path is None:
        summary = _safe_stop_summary(config, "local VideoMAE model/cache is missing", env_guard)
        summary.update({"model_missing": True, "clip_export_summary": clip_summary})
        return _write_token_summary(summary, token_summary_path, token_summary_md_path)

    encoder_cfg = _encoder_config(config, model_path)
    encoder = build_frozen_video_encoder(encoder_cfg)
    if not encoder.is_available():
        summary = _safe_stop_summary(
            config,
            f"local VideoMAE unavailable: {encoder.availability.get('reason')}",
            env_guard,
        )
        summary.update(
            {
                "model_missing": True,
                "model_path": model_path,
                "encoder_availability": encoder.availability,
                "clip_export_summary": clip_summary,
            }
        )
        return _write_token_summary(summary, token_summary_path, token_summary_md_path)

    token_smoke_dir.mkdir(parents=True, exist_ok=True)
    for old in token_smoke_dir.glob("*.pt"):
        old.unlink()
    if token_manifest_path.exists():
        token_manifest_path.unlink()

    records: list[dict[str, Any]] = []
    frame_repeat = int(config["videomae"].get("frame_repeat", 4))
    batch_size = int(config["dry_run_limits"].get("batch_size", 1))
    image_size = int(config["clip_export"]["image_size"])
    with torch.no_grad():
        for clip_path in clip_summary.get("clip_cache_files", []):
            sample = load_clip_cache_npz(clip_path)
            if not validate_clip_cache(
                sample,
                expected_context=int(config["window"]["context_len"]),
                expected_current=int(config["window"]["current_len"]),
                expected_future=int(config["window"]["future_len"]),
            ):
                raise ValueError(f"Invalid Step24 clip cache: {clip_path}")
            videos = clip_cache_to_torch_videos(sample, image_size=image_size)
            context_tokens = _encode_frame_stack(encoder, videos["context_video"], frame_repeat, batch_size)
            current_tokens = _encode_frame_stack(encoder, videos["current_video"], frame_repeat, batch_size)
            future_tokens = _encode_frame_stack(encoder, videos["future_video"], frame_repeat, batch_size)
            metadata = dict(videos["metadata"])
            sample_id = str(metadata.get("sample_id") or Path(clip_path).stem)
            artifact_path = token_smoke_dir / f"{sample_id}.pt"
            torch.save(
                {
                    "schema_version": "0.1.0",
                    "stage": "bridgedata_v2_tfds_token_extraction_step24",
                    "sample_id": sample_id,
                    "trajectory_id": metadata.get("trajectory_id"),
                    "image_field": metadata.get("image_field"),
                    "context_tokens": context_tokens.to(dtype=torch.float16).cpu(),
                    "current_tokens": current_tokens.to(dtype=torch.float16).cpu(),
                    "future_tokens": future_tokens.to(dtype=torch.float16).cpu(),
                    "metadata": metadata,
                    "encoder_availability": encoder.availability,
                },
                artifact_path,
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "trajectory_id": metadata.get("trajectory_id"),
                    "context_token_shape": list(context_tokens.shape),
                    "current_token_shape": list(current_tokens.shape),
                    "future_token_shape": list(future_tokens.shape),
                    "token_artifact_path": str(artifact_path),
                    "image_field": metadata.get("image_field"),
                    "action_used_as_input": False,
                    "language_used_as_input": False,
                    "goal_used_as_input": False,
                }
            )

    write_token_manifest_jsonl(records, token_manifest_path)
    manifest_summary = summarize_token_manifest(records)
    summary = {
        "stage": "bridgedata_v2_tfds_token_extraction_step24",
        "token_extraction_performed": bool(records),
        "safe_stop": not bool(records),
        "reason": None if records else "No token records were generated.",
        "model_missing": False,
        "num_windows_tokenized": manifest_summary["num_windows_tokenized"],
        "num_token_artifacts": manifest_summary["num_token_artifacts"],
        "context_token_shape_example": manifest_summary["context_token_shape_example"],
        "current_token_shape_example": manifest_summary["current_token_shape_example"],
        "future_token_shape_example": manifest_summary["future_token_shape_example"],
        "model_loaded_local_only": True,
        "model_path": model_path,
        "model_download_performed": False,
        "training_performed": False,
        "importance_generation_performed": False,
        "large_token_shards_generated": False,
        "data_token_shards_written": False,
        "image_field": config["resolved_fields"]["image_field"],
        "image_field_is_metadata_flag": False,
        "token_manifest_jsonl": str(token_manifest_path),
        "token_smoke_dir": str(token_smoke_dir),
        "encoder_availability": encoder.availability,
        "env_isaaclab_guard": env_guard,
        "clip_export_summary": clip_summary,
        "safety_gate_pass": True,
    }
    return _write_token_summary(summary, token_summary_path, token_summary_md_path)


def _encode_frame_stack(encoder: Any, video: torch.Tensor, frame_repeat: int, batch_size: int) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, int(video.shape[0]), batch_size):
        frames = video[start : start + batch_size]
        repeated = frames.unsqueeze(1).repeat(1, frame_repeat, 1, 1, 1)
        outputs.append(encoder.encode(repeated))
    return torch.cat(outputs, dim=0).contiguous()


def _encoder_config(config: dict[str, Any], model_path: str) -> dict[str, Any]:
    video_cfg = config["videomae"]
    return {
        "name": "videomae",
        "model_name_or_path": model_path,
        "allow_download": False,
        "local_files_only": True,
        "image_size": int(video_cfg.get("image_size", 224)),
        "num_frames": int(video_cfg.get("frame_repeat", 4)),
        "output_mode": str(video_cfg.get("output_mode", "last_hidden_state")),
        "device": str(video_cfg.get("device", "cuda_if_available")),
        "ignore_mismatched_sizes": True,
    }


def _find_local_model_path(candidates: list[str]) -> str | None:
    for candidate in candidates:
        path = Path(str(candidate)).expanduser()
        if path.exists() and (path / "config.json").exists():
            return str(path)
    return None


def _env_isaaclab_guard() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def _safe_stop_summary(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    return {
        "stage": "bridgedata_v2_tfds_token_extraction_step24",
        "token_extraction_performed": False,
        "safe_stop": True,
        "reason": reason,
        "model_missing": "model" in reason.lower() or "videomae" in reason.lower(),
        "num_windows_tokenized": 0,
        "num_token_artifacts": 0,
        "context_token_shape_example": None,
        "current_token_shape_example": None,
        "future_token_shape_example": None,
        "model_loaded_local_only": False,
        "model_download_performed": False,
        "training_performed": False,
        "importance_generation_performed": False,
        "large_token_shards_generated": False,
        "data_token_shards_written": False,
        "image_field": config["resolved_fields"]["image_field"],
        "image_field_is_metadata_flag": False,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }


def _write_token_summary(summary: dict[str, Any], json_path: Path, md_path: Path) -> dict[str, Any]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# BridgeData V2 TFDS Token Extraction Summary",
        "",
        f"- token_extraction_performed: `{str(summary.get('token_extraction_performed', False)).lower()}`",
        f"- safe_stop: `{str(summary.get('safe_stop', False)).lower()}`",
        f"- model_missing: `{str(summary.get('model_missing', False)).lower()}`",
        f"- image_field: `{summary.get('image_field')}`",
        f"- num_windows_tokenized: `{summary.get('num_windows_tokenized')}`",
        f"- context_token_shape_example: `{summary.get('context_token_shape_example')}`",
        f"- current_token_shape_example: `{summary.get('current_token_shape_example')}`",
        f"- future_token_shape_example: `{summary.get('future_token_shape_example')}`",
        f"- model_download_performed: `{str(summary.get('model_download_performed', False)).lower()}`",
        f"- training_performed: `{str(summary.get('training_performed', False)).lower()}`",
        f"- importance_generation_performed: `{str(summary.get('importance_generation_performed', False)).lower()}`",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


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
    summary = extract_bridgedata_v2_tfds_tokens_from_config(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
