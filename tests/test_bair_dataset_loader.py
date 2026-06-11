"""Tests for BAIRRobotPushingDataset."""

from __future__ import annotations

from pathlib import Path

import torch

from data.bair_dataset import BAIRRobotPushingDataset
from data.real_video_index import write_metadata_jsonl


def _write_clip(root: Path, split: str, index: int) -> dict[str, object]:
    split_dir = root / split
    clips_dir = split_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    sample_id = f"bair_{split}_{index:06d}"
    clip_name = f"{sample_id}.pt"
    payload = {
        "video": torch.rand(8, 3, 24, 32),
        "task_text": "predict robot pushing future visual dynamics",
        "sample_id": sample_id,
        "fps": 5,
        "source": "bair_robot_pushing_small",
        "split": split,
        "camera": "image_main",
        "actions": torch.ones(8, 4),
        "endeffector_pos": torch.ones(8, 3),
        "metadata": {"source": "bair_robot_pushing_small", "split": split},
    }
    torch.save(payload, clips_dir / clip_name)
    return {
        "sample_id": sample_id,
        "source_type": "pt_clip",
        "clip_path": f"clips/{clip_name}",
        "task_text": payload["task_text"],
        "num_frames": 8,
        "fps": 5,
        "height": 224,
        "width": 224,
        "source": "bair_robot_pushing_small",
        "split": split,
        "camera": "image_main",
        "has_action": True,
        "has_endeffector_pos": True,
    }


def test_bair_dataset_loader_reads_exported_pt_clips(tmp_path: Path) -> None:
    root = tmp_path / "bair_robot_pushing_small_subset"
    records = [_write_clip(root, "train", index) for index in range(2)]
    write_metadata_jsonl(root / "train" / "metadata.jsonl", records)

    dataset = BAIRRobotPushingDataset(root=root, split="train", past_len=4, future_len=4, image_size=64)
    sample = dataset[0]

    assert len(dataset) == 2
    assert list(sample["past_video"].shape) == [4, 3, 64, 64]
    assert list(sample["future_video"].shape) == [4, 3, 64, 64]
    assert list(sample["actions"].shape) == [8, 4]
    assert list(sample["endeffector_pos"].shape) == [8, 3]
    assert sample["sample_id"] == "bair_train_000000"
    assert sample["metadata"]["source"] == "bair_robot_pushing_small"
