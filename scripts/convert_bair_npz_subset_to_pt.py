"""Convert raw BAIR NumPy clips into the project .pt clip subset schema."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.real_video_transforms import ensure_video_tensor, resize_video_tensor, validate_video_tensor


TASK_TEXT = "predict robot pushing future visual dynamics"
SOURCE = "bair_robot_pushing_small"
DEFAULT_FPS = 5


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _metadata_for_clip(payload: dict[str, Any], clip_path: str) -> dict[str, Any]:
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


def _to_optional_tensor(value: np.ndarray, enabled: bool) -> torch.Tensor | None:
    if not enabled or value.size == 0:
        return None
    return torch.as_tensor(value, dtype=torch.float32).contiguous()


def convert_bair_npz_subset_to_pt(config: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    project_root = Path(config.get("project_root", "."))
    subset_cfg = config["subset"]
    output_root = Path(subset_cfg["output_root"])
    raw_root = output_root / "raw_npz"
    image_size = int(subset_cfg.get("image_size", 224))
    total_frames = int(subset_cfg.get("total_frames", 8))
    split_names = list(subset_cfg.get("split_names", ["train", "test"]))
    summary: dict[str, Any] = {
        "export_success": False,
        "blocking_reason": None,
        "data_dir": config["input"]["data_dir"],
        "output_root": str(output_root),
        "raw_output_root": str(raw_root),
        "splits": {},
        "sample_shapes": {},
    }
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing raw NPZ export root: {raw_root}")

    for split in split_names:
        raw_split = raw_root / split
        split_dir = output_root / split
        clips_dir = split_dir / "clips"
        if split_dir.exists():
            shutil.rmtree(split_dir)
        clips_dir.mkdir(parents=True, exist_ok=True)
        raw_records = _read_jsonl(raw_split / "metadata.jsonl")
        metadata: list[dict[str, Any]] = []
        for record in raw_records:
            sample_id = str(record["sample_id"])
            raw_path = raw_split / record["raw_npz_path"]
            with np.load(raw_path) as npz:
                video = ensure_video_tensor(npz["video"])
                video = resize_video_tensor(video[:total_frames], image_size=image_size)
                validate_video_tensor(video, min_frames=total_frames)
                payload: dict[str, Any] = {
                    "video": video,
                    "task_text": TASK_TEXT,
                    "sample_id": sample_id,
                    "fps": DEFAULT_FPS,
                    "source": SOURCE,
                    "split": split,
                    "camera": record.get("camera", "image_main"),
                    "metadata": {
                        "source": SOURCE,
                        "split": split,
                        "camera": record.get("camera", "image_main"),
                        "original_num_frames": int(record.get("original_num_frames", total_frames)),
                        "num_frames": int(video.shape[0]),
                        "height": int(video.shape[2]),
                        "width": int(video.shape[3]),
                    },
                }
                actions = _to_optional_tensor(npz["actions"], bool(npz["has_action"]))
                endeffector_pos = _to_optional_tensor(npz["endeffector_pos"], bool(npz["has_endeffector_pos"]))
                if actions is not None:
                    payload["actions"] = actions
                if endeffector_pos is not None:
                    payload["endeffector_pos"] = endeffector_pos
            clip_name = f"{sample_id}.pt"
            torch.save(payload, clips_dir / clip_name)
            metadata.append(_metadata_for_clip(payload, f"clips/{clip_name}"))
            if split not in summary["sample_shapes"]:
                summary["sample_shapes"][split] = {
                    "video": list(payload["video"].shape),
                    "past": [4, 3, image_size, image_size],
                    "future": [4, 3, image_size, image_size],
                    "actions": list(payload["actions"].shape) if isinstance(payload.get("actions"), torch.Tensor) else None,
                    "endeffector_pos": (
                        list(payload["endeffector_pos"].shape)
                        if isinstance(payload.get("endeffector_pos"), torch.Tensor)
                        else None
                    ),
                }
        _write_jsonl(split_dir / "metadata.jsonl", metadata)
        requested = int(subset_cfg.get(f"{split}_max_episodes", len(metadata)))
        summary["splits"][split] = {
            "requested": requested,
            "exported": len(metadata),
            "metadata_path": str(split_dir / "metadata.jsonl"),
            "clips_dir": str(clips_dir),
        }

    summary["export_success"] = all(
        int(summary["splits"].get(split, {}).get("exported", 0))
        == int(subset_cfg.get(f"{split}_max_episodes", 0))
        for split in split_names
    )
    summary["elapsed_time_sec"] = time.time() - start
    summary_path = output_root / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path = output_root / "export_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# BAIR Subset Export Report\n\n"
        f"- export_success: `{str(summary['export_success']).lower()}`\n"
        f"- output_root: `{summary['output_root']}`\n"
        f"- train_exported: `{summary['splits'].get('train', {}).get('exported')}`\n"
        f"- test_exported: `{summary['splits'].get('test', {}).get('exported')}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"BAIR_SUBSET_EXPORT_PASS = {str(summary['export_success']).lower()}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_bair_npz_subset_to_pt(load_yaml(args.config))
    if not summary["export_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
