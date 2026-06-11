"""Export BAIR TFDS examples to raw NumPy clips without importing torch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


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


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _extract_fields(example: dict[str, Any], camera: str) -> dict[str, np.ndarray | None]:
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
        "video": _to_numpy(video),
        "actions": _to_numpy(actions),
        "endeffector_pos": _to_numpy(endeffector_pos),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _optional_sequence(value: np.ndarray | None, total_frames: int) -> tuple[np.ndarray, bool]:
    if value is None or value.ndim == 0 or value.shape[0] < total_frames:
        return np.zeros((0,), dtype=np.float32), False
    return value[:total_frames].astype(np.float32), True


def export_bair_subset_npz(config: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_datasets as tfds

    start = time.time()
    input_cfg = config["input"]
    subset_cfg = config["subset"]
    output_root = Path(subset_cfg["output_root"])
    raw_root = output_root / "raw_npz"
    data_dir = Path(input_cfg["data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing BAIR TFDS data_dir: {data_dir}")

    split_limits = {
        "train": int(subset_cfg.get("train_max_episodes", 500)),
        "test": int(subset_cfg.get("test_max_episodes", 64)),
    }
    split_names = list(subset_cfg.get("split_names", ["train", "test"]))
    total_frames = int(subset_cfg.get("total_frames", 8))
    camera = str(subset_cfg.get("camera", "image_main"))
    raw_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "stage": "bair_robot_pushing_small_subset_500_64_raw_npz",
        "tfds_name": input_cfg.get("tfds_name", "bair_robot_pushing_small"),
        "tfds_version": input_cfg.get("tfds_version", "2.0.0"),
        "data_dir": str(data_dir),
        "raw_output_root": str(raw_root),
        "splits": {},
    }
    for split in split_names:
        split_dir = raw_root / split
        clips_dir = split_dir / "clips"
        if split_dir.exists():
            import shutil

            shutil.rmtree(split_dir)
        clips_dir.mkdir(parents=True, exist_ok=True)
        metadata: list[dict[str, Any]] = []
        skipped_short = 0
        dataset = tfds.load(
            f"{summary['tfds_name']}:{summary['tfds_version']}",
            data_dir=str(data_dir),
            split=split,
            shuffle_files=False,
            download=False,
        )
        for example in tfds.as_numpy(dataset):
            if len(metadata) >= split_limits[split]:
                break
            fields = _extract_fields(example, camera=camera)
            video = fields["video"]
            if video is None or video.shape[0] < total_frames:
                skipped_short += 1
                continue
            sample_id = f"bair_{split}_{len(metadata):06d}"
            actions, has_action = _optional_sequence(fields["actions"], total_frames)
            endeffector_pos, has_endeffector_pos = _optional_sequence(fields["endeffector_pos"], total_frames)
            clip_name = f"{sample_id}.npz"
            np.savez_compressed(
                clips_dir / clip_name,
                video=video[:total_frames],
                actions=actions,
                endeffector_pos=endeffector_pos,
                has_action=np.asarray(has_action),
                has_endeffector_pos=np.asarray(has_endeffector_pos),
            )
            metadata.append(
                {
                    "sample_id": sample_id,
                    "raw_npz_path": f"clips/{clip_name}",
                    "source": "bair_robot_pushing_small",
                    "split": split,
                    "camera": camera,
                    "num_frames": total_frames,
                    "original_num_frames": int(video.shape[0]),
                    "has_action": has_action,
                    "has_endeffector_pos": has_endeffector_pos,
                }
            )
        _write_jsonl(split_dir / "metadata.jsonl", metadata)
        summary["splits"][split] = {
            "requested": split_limits[split],
            "exported": len(metadata),
            "skipped_short": skipped_short,
            "metadata_path": str(split_dir / "metadata.jsonl"),
            "clips_dir": str(clips_dir),
        }

    summary["export_success"] = all(
        int(summary["splits"].get(split, {}).get("exported", 0)) == split_limits[split] for split in split_names
    )
    summary["elapsed_time_sec"] = time.time() - start
    output_path = raw_root / "raw_npz_export_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"BAIR_RAW_NPZ_EXPORT_PASS = {str(summary['export_success']).lower()}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_bair_subset_npz(load_yaml(args.config))
    if not summary["export_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
