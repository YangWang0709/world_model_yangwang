"""Export BAIR context/current/future raw NumPy windows without importing torch."""

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
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


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
    return {"video": _to_numpy(video), "actions": _to_numpy(actions), "endeffector_pos": _to_numpy(endeffector_pos)}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _optional_sequence(value: np.ndarray | None, total_frames: int) -> tuple[np.ndarray, bool]:
    if value is None or value.ndim == 0 or value.shape[0] < total_frames:
        return np.zeros((0,), dtype=np.float32), False
    return value[:total_frames].astype(np.float32), True


def _split_complete(raw_root: Path, split: str, expected: int) -> bool:
    metadata_path = raw_root / split / "metadata.jsonl"
    clips_dir = raw_root / split / "clips"
    if not metadata_path.exists() or not clips_dir.exists():
        return False
    count = sum(1 for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return count == int(expected)


def export_bair_context_windows(config_path: str | Path) -> dict[str, Any]:
    import tensorflow_datasets as tfds  # type: ignore

    config = load_yaml(config_path)
    start = time.time()
    input_cfg = config["input"]
    dataset_cfg = config["dataset"]
    output_root = Path(dataset_cfg["output_root"])
    raw_root = output_root / "raw_npz"
    raw_root.mkdir(parents=True, exist_ok=True)
    split_limits = {"train": int(dataset_cfg["train_samples"]), "test": int(dataset_cfg["test_samples"])}
    if not bool(dataset_cfg.get("overwrite", False)) and all(_split_complete(raw_root, split, expected) for split, expected in split_limits.items()):
        summary_path = raw_root / "raw_npz_export_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["skipped_existing"] = True
            print(json.dumps(summary, indent=2))
            return summary
    context_len = int(dataset_cfg["context_len"])
    current_len = int(dataset_cfg["current_len"])
    future_len = int(dataset_cfg["future_len"])
    total_frames = context_len + current_len + future_len
    camera = str(dataset_cfg.get("camera", "image_main"))
    summary: dict[str, Any] = {
        "stage": "bair_context_windows_1000_128_raw_npz",
        "tfds_name": input_cfg.get("tfds_name", "bair_robot_pushing_small"),
        "tfds_version": input_cfg.get("tfds_version", "0.1.0"),
        "data_dir": str(input_cfg["data_dir"]),
        "raw_output_root": str(raw_root),
        "context_len": context_len,
        "current_len": current_len,
        "future_len": future_len,
        "total_frames": total_frames,
        "use_action_as_input": False,
        "use_endeffector_as_input": False,
        "splits": {},
    }
    for split in dataset_cfg.get("split_names", ["train", "test"]):
        split = str(split)
        split_dir = raw_root / split
        clips_dir = split_dir / "clips"
        if split_dir.exists() and bool(dataset_cfg.get("overwrite", False)):
            import shutil

            shutil.rmtree(split_dir)
        clips_dir.mkdir(parents=True, exist_ok=True)
        metadata: list[dict[str, Any]] = []
        skipped_short = 0
        dataset = tfds.load(
            f"{summary['tfds_name']}:{summary['tfds_version']}",
            data_dir=str(input_cfg["data_dir"]),
            split=split,
            shuffle_files=False,
            download=False,
        )
        for example in tfds.as_numpy(dataset):
            if len(metadata) >= split_limits[split]:
                break
            fields = _extract_fields(example, camera)
            video = fields["video"]
            if video is None or video.shape[0] < total_frames:
                skipped_short += 1
                continue
            sample_id = f"bair_context_{split}_{len(metadata):06d}"
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
                    "context_len": context_len,
                    "current_len": current_len,
                    "future_len": future_len,
                    "total_frames": total_frames,
                    "original_num_frames": int(video.shape[0]),
                    "has_action": bool(has_action),
                    "has_endeffector_pos": bool(has_endeffector_pos),
                    "use_action_as_input": False,
                    "use_endeffector_as_input": False,
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
    summary["export_success"] = all(int(summary["splits"][split]["exported"]) == split_limits[split] for split in split_limits)
    summary["elapsed_time_sec"] = round(time.time() - start, 3)
    (raw_root / "raw_npz_export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"BAIR_CONTEXT_RAW_NPZ_EXPORT_PASS = {str(summary['export_success']).lower()}")
    if not summary["export_success"]:
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    export_bair_context_windows(parse_args().config)


if __name__ == "__main__":
    main()
