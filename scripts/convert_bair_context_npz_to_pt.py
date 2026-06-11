"""Convert raw BAIR context-window NPZ files to .pt context/current/future clips."""

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
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


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


def _optional_tensor(value: np.ndarray, enabled: bool) -> torch.Tensor | None:
    if not enabled or value.size == 0:
        return None
    return torch.as_tensor(value, dtype=torch.float32).contiguous()


def _metadata_for_clip(payload: dict[str, Any], clip_path: str) -> dict[str, Any]:
    return {
        "sample_id": str(payload["sample_id"]),
        "source_type": "pt_context_window",
        "clip_path": clip_path,
        "path": clip_path,
        "task_text": str(payload["task_text"]),
        "context_len": int(payload["context_video"].shape[0]),
        "current_len": int(payload["current_video"].shape[0]),
        "future_len": int(payload["future_video"].shape[0]),
        "total_frames": int(payload["metadata"]["total_frames"]),
        "fps": int(payload.get("fps", DEFAULT_FPS)),
        "height": int(payload["context_video"].shape[2]),
        "width": int(payload["context_video"].shape[3]),
        "source": SOURCE,
        "split": str(payload["split"]),
        "camera": str(payload.get("camera", "image_main")),
        "has_action": isinstance(payload.get("actions"), torch.Tensor),
        "has_endeffector_pos": isinstance(payload.get("endeffector_pos"), torch.Tensor),
        "use_action_as_input": False,
        "use_endeffector_as_input": False,
    }


def convert_bair_context_npz_to_pt(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    start = time.time()
    dataset_cfg = config["dataset"]
    output_root = Path(dataset_cfg["output_root"])
    raw_root = output_root / "raw_npz"
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing raw NPZ export root: {raw_root}")
    image_size = int(dataset_cfg.get("image_size", 224))
    context_len = int(dataset_cfg["context_len"])
    current_len = int(dataset_cfg["current_len"])
    future_len = int(dataset_cfg["future_len"])
    total_frames = context_len + current_len + future_len
    summary: dict[str, Any] = {
        "export_success": False,
        "output_root": str(output_root),
        "raw_output_root": str(raw_root),
        "context_len": context_len,
        "current_len": current_len,
        "future_len": future_len,
        "total_frames": total_frames,
        "use_action_as_input": False,
        "use_endeffector_as_input": False,
        "splits": {},
        "sample_shapes": {},
    }
    for split in dataset_cfg.get("split_names", ["train", "test"]):
        split = str(split)
        raw_split = raw_root / split
        split_dir = output_root / split
        clips_dir = split_dir / "clips"
        if split_dir.exists():
            shutil.rmtree(split_dir)
        clips_dir.mkdir(parents=True, exist_ok=True)
        metadata = []
        for record in _read_jsonl(raw_split / "metadata.jsonl"):
            sample_id = str(record["sample_id"])
            with np.load(raw_split / record["raw_npz_path"]) as npz:
                video = resize_video_tensor(ensure_video_tensor(npz["video"])[:total_frames], image_size)
                validate_video_tensor(video, min_frames=total_frames)
                context_end = context_len
                current_end = context_len + current_len
                payload: dict[str, Any] = {
                    "context_video": video[:context_end].contiguous(),
                    "current_video": video[context_end:current_end].contiguous(),
                    "future_video": video[current_end:total_frames].contiguous(),
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
                        "total_frames": total_frames,
                        "context_len": context_len,
                        "current_len": current_len,
                        "future_len": future_len,
                        "height": int(video.shape[2]),
                        "width": int(video.shape[3]),
                        "action_used_as_input": False,
                        "endeffector_used_as_input": False,
                    },
                }
                actions = _optional_tensor(npz["actions"], bool(npz["has_action"]))
                endeffector_pos = _optional_tensor(npz["endeffector_pos"], bool(npz["has_endeffector_pos"]))
                if actions is not None:
                    payload["actions"] = actions
                if endeffector_pos is not None:
                    payload["endeffector_pos"] = endeffector_pos
            clip_name = f"{sample_id}.pt"
            torch.save(payload, clips_dir / clip_name)
            metadata.append(_metadata_for_clip(payload, f"clips/{clip_name}"))
            summary["sample_shapes"].setdefault(
                split,
                {
                    "context_video": list(payload["context_video"].shape),
                    "current_video": list(payload["current_video"].shape),
                    "future_video": list(payload["future_video"].shape),
                    "actions": list(payload["actions"].shape) if isinstance(payload.get("actions"), torch.Tensor) else None,
                    "endeffector_pos": list(payload["endeffector_pos"].shape) if isinstance(payload.get("endeffector_pos"), torch.Tensor) else None,
                },
            )
        _write_jsonl(split_dir / "metadata.jsonl", metadata)
        requested = int(dataset_cfg[f"{split}_samples"])
        summary["splits"][split] = {
            "requested": requested,
            "exported": len(metadata),
            "metadata_path": str(split_dir / "metadata.jsonl"),
            "clips_dir": str(clips_dir),
        }
    summary["export_success"] = all(
        int(summary["splits"][split]["exported"]) == int(dataset_cfg[f"{split}_samples"])
        for split in ("train", "test")
    )
    summary["elapsed_time_sec"] = round(time.time() - start, 3)
    (output_root / "export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "export_report.md").write_text(
        "# BAIR Context Window Export Report\n\n"
        f"- export_success: `{str(summary['export_success']).lower()}`\n"
        f"- train_exported: `{summary['splits'].get('train', {}).get('exported')}`\n"
        f"- test_exported: `{summary['splits'].get('test', {}).get('exported')}`\n"
        f"- context/current/future: `{context_len}` / `{current_len}` / `{future_len}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"BAIR_CONTEXT_WINDOW_EXPORT_PASS = {str(summary['export_success']).lower()}")
    if not summary["export_success"]:
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    convert_bair_context_npz_to_pt(parse_args().config)


if __name__ == "__main__":
    main()
