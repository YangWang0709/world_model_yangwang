"""Export a tiny BAIR Robot Pushing subset as project-local .pt clips."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from .bair_tfds_utils import BAIR_TFDS_NAME, BAIR_TFDS_VERSION, check_tfds_available
from .real_video_index import write_metadata_jsonl
from .real_video_transforms import ensure_video_tensor, resize_video_tensor, validate_video_tensor


TASK_TEXT = "predict robot pushing future visual dynamics"
SOURCE = "bair_robot_pushing_small"
DEFAULT_FPS = 5


def _to_plain(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def _lookup_nested(mapping: Any, paths: Iterable[tuple[str, ...]]) -> Any | None:
    for path in paths:
        current = mapping
        found = True
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                found = False
                break
        if found:
            return current
    return None


def extract_bair_fields(example: dict[str, Any], camera: str = "image_main") -> dict[str, Any]:
    """Extract video/action fields from common BAIR TFDS structures."""

    video = _lookup_nested(
        example,
        (
            ("steps", "observation", camera),
            ("steps", camera),
            ("observation", camera),
            (camera,),
            ("image_main",),
            ("image",),
        ),
    )
    actions = _lookup_nested(example, (("steps", "action"), ("action",)))
    endeffector_pos = _lookup_nested(
        example,
        (
            ("steps", "observation", "endeffector_pos"),
            ("steps", "endeffector_pos"),
            ("observation", "endeffector_pos"),
            ("endeffector_pos",),
            ("steps", "observation", "state"),
            ("observation", "state"),
            ("state",),
        ),
    )
    if video is None:
        raise KeyError(f"Could not find BAIR camera field {camera!r}")
    return {
        "video": _to_plain(video),
        "actions": _to_plain(actions) if actions is not None else None,
        "endeffector_pos": _to_plain(endeffector_pos) if endeffector_pos is not None else None,
    }


def _optional_sequence_tensor(value: Any, total_frames: int) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim == 0:
        return None
    if tensor.shape[0] < total_frames:
        return None
    return tensor[:total_frames].to(torch.float32).contiguous()


def convert_bair_sequence_to_clip(
    example: dict[str, Any],
    sample_id: str,
    split: str,
    camera: str = "image_main",
    total_frames: int = 8,
    image_size: int = 224,
    include_action: bool = True,
    include_endeffector_pos: bool = True,
    fps: int = DEFAULT_FPS,
    task_text: str = TASK_TEXT,
) -> dict[str, Any] | None:
    """Convert one BAIR-like example into the project .pt clip schema."""

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    fields = extract_bair_fields(example, camera=camera)
    video = ensure_video_tensor(fields["video"])
    if video.shape[0] < total_frames:
        return None
    video = resize_video_tensor(video[:total_frames], image_size)
    validate_video_tensor(video, min_frames=total_frames)

    payload: dict[str, Any] = {
        "video": video,
        "task_text": task_text,
        "sample_id": sample_id,
        "fps": int(fps),
        "source": SOURCE,
        "split": split,
        "camera": camera,
        "metadata": {
            "source": SOURCE,
            "split": split,
            "camera": camera,
            "original_num_frames": int(ensure_video_tensor(fields["video"]).shape[0]),
            "num_frames": int(video.shape[0]),
            "height": int(video.shape[2]),
            "width": int(video.shape[3]),
        },
    }
    if include_action:
        payload["actions"] = _optional_sequence_tensor(fields.get("actions"), total_frames)
    if include_endeffector_pos:
        payload["endeffector_pos"] = _optional_sequence_tensor(fields.get("endeffector_pos"), total_frames)
    return payload


def metadata_for_clip(payload: dict[str, Any], clip_path: str) -> dict[str, Any]:
    video = payload["video"]
    return {
        "sample_id": str(payload["sample_id"]),
        "source_type": "pt_clip",
        "clip_path": clip_path,
        "path": clip_path,
        "task_text": str(payload["task_text"]),
        "num_frames": int(video.shape[0]),
        "fps": int(payload.get("fps", DEFAULT_FPS)),
        "height": int(video.shape[2]),
        "width": int(video.shape[3]),
        "source": SOURCE,
        "split": str(payload["split"]),
        "camera": str(payload.get("camera", "image_main")),
        "has_action": isinstance(payload.get("actions"), torch.Tensor),
        "has_endeffector_pos": isinstance(payload.get("endeffector_pos"), torch.Tensor),
    }


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def _safe_clear_split_dir(split_dir: Path, output_root: Path) -> None:
    resolved = split_dir.resolve()
    root = output_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Refusing to clear unsafe split dir: {split_dir}")
    if split_dir.exists():
        shutil.rmtree(split_dir)


def _write_export_report(report_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# BAIR Subset Export Report",
        "",
        f"- export_success: `{str(summary.get('export_success', False)).lower()}`",
        f"- blocking_reason: `{summary.get('blocking_reason')}`",
        f"- output_root: `{summary.get('output_root')}`",
        f"- train_exported: `{summary.get('splits', {}).get('train', {}).get('exported', 0)}`",
        f"- test_exported: `{summary.get('splits', {}).get('test', {}).get('exported', 0)}`",
        f"- elapsed_time_sec: `{summary.get('elapsed_time_sec')}`",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_bair_subset(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Stream a small BAIR subset from TFDS into .pt clips."""

    if not isinstance(config, dict):
        config = _load_yaml(config)
    start = time.time()
    project_root = Path(config.get("project_root", "."))
    input_cfg = config["input"]
    subset_cfg = config["subset"]
    output_root = Path(subset_cfg["output_root"])
    summary_path = output_root / "export_summary.json"
    report_path = project_root / "docs" / "BAIR_SUBSET_EXPORT_REPORT.md"
    caps = check_tfds_available()
    summary: dict[str, Any] = {
        "export_success": False,
        "blocking_reason": None,
        "tfds_name": input_cfg.get("tfds_name", BAIR_TFDS_NAME),
        "tfds_version": input_cfg.get("tfds_version", BAIR_TFDS_VERSION),
        "data_dir": input_cfg["data_dir"],
        "output_root": str(output_root),
        "splits": {},
        "capability_summary": caps,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    if not caps["tensorflow_datasets_available"]:
        summary["blocking_reason"] = "tensorflow_datasets is not available"
        summary["elapsed_time_sec"] = time.time() - start
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        _write_export_report(report_path, summary)
        return summary

    try:
        import tensorflow_datasets as tfds
    except Exception as exc:  # pragma: no cover - handled in capability normally
        summary["blocking_reason"] = f"tensorflow_datasets import failed: {exc}"
        summary["elapsed_time_sec"] = time.time() - start
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        _write_export_report(report_path, summary)
        return summary

    split_limits = {
        "train": int(subset_cfg.get("train_max_episodes", 100)),
        "test": int(subset_cfg.get("test_max_episodes", 16)),
    }
    camera = str(subset_cfg.get("camera", "image_main"))
    total_frames = int(subset_cfg.get("total_frames", 8))
    image_size = int(subset_cfg.get("image_size", 224))
    include_action = bool(subset_cfg.get("include_action", True))
    include_endeffector_pos = bool(subset_cfg.get("include_endeffector_pos", True))
    split_names = list(subset_cfg.get("split_names", ["train", "test"]))

    try:
        for split in split_names:
            limit = split_limits.get(split, 0)
            split_dir = output_root / split
            clips_dir = split_dir / "clips"
            _safe_clear_split_dir(split_dir, output_root)
            clips_dir.mkdir(parents=True, exist_ok=True)
            metadata: list[dict[str, Any]] = []
            skipped_short = 0
            dataset = tfds.load(
                f"{input_cfg.get('tfds_name', BAIR_TFDS_NAME)}:{input_cfg.get('tfds_version', BAIR_TFDS_VERSION)}",
                data_dir=str(input_cfg["data_dir"]),
                split=split,
                shuffle_files=False,
                download=False,
            )
            for raw_index, example in enumerate(tfds.as_numpy(dataset)):
                if len(metadata) >= limit:
                    break
                sample_id = f"bair_{split}_{len(metadata):06d}"
                payload = convert_bair_sequence_to_clip(
                    example,
                    sample_id=sample_id,
                    split=split,
                    camera=camera,
                    total_frames=total_frames,
                    image_size=image_size,
                    include_action=include_action,
                    include_endeffector_pos=include_endeffector_pos,
                )
                if payload is None:
                    skipped_short += 1
                    continue
                clip_name = f"{sample_id}.pt"
                torch.save(payload, clips_dir / clip_name)
                metadata.append(metadata_for_clip(payload, f"clips/{clip_name}"))
            write_metadata_jsonl(split_dir / "metadata.jsonl", metadata)
            summary["splits"][split] = {
                "requested": limit,
                "exported": len(metadata),
                "skipped_short": skipped_short,
                "metadata_path": str(split_dir / "metadata.jsonl"),
                "clips_dir": str(clips_dir),
            }
        summary["export_success"] = all(
            int(summary["splits"].get(split, {}).get("exported", 0)) > 0 for split in split_names
        )
    except Exception as exc:  # pragma: no cover - depends on TFDS/network/dataset availability
        summary["blocking_reason"] = str(exc)
    summary["elapsed_time_sec"] = time.time() - start
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_export_report(report_path, summary)
    return summary
