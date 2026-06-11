"""Schema tests for BAIR subset export helpers without real TFDS data."""

from __future__ import annotations

import numpy as np
import torch

from data.bair_subset_export import convert_bair_sequence_to_clip, metadata_for_clip


def test_convert_bair_sequence_to_clip_schema() -> None:
    rng = np.random.default_rng(42)
    example = {
        "steps": {
            "observation": {
                "image_main": rng.integers(0, 256, size=(10, 64, 64, 3), dtype=np.uint8),
                "endeffector_pos": np.ones((10, 3), dtype=np.float32),
            },
            "action": np.ones((10, 4), dtype=np.float32),
        }
    }

    payload = convert_bair_sequence_to_clip(
        example,
        sample_id="bair_train_000000",
        split="train",
        total_frames=8,
        image_size=224,
    )

    assert payload is not None
    video = payload["video"]
    assert isinstance(video, torch.Tensor)
    assert list(video.shape) == [8, 3, 224, 224]
    assert video.dtype == torch.float32
    assert float(video.min()) >= 0.0
    assert float(video.max()) <= 1.0
    assert list(payload["actions"].shape) == [8, 4]
    assert list(payload["endeffector_pos"].shape) == [8, 3]
    assert payload["metadata"]["source"] == "bair_robot_pushing_small"

    metadata = metadata_for_clip(payload, "clips/bair_train_000000.pt")
    assert metadata["sample_id"] == "bair_train_000000"
    assert metadata["source_type"] == "pt_clip"
    assert metadata["num_frames"] == 8
    assert metadata["height"] == 224
    assert metadata["width"] == 224
    assert metadata["has_action"] is True
    assert metadata["has_endeffector_pos"] is True


def test_convert_bair_sequence_skips_short_sequence() -> None:
    example = {"image_main": np.zeros((4, 64, 64, 3), dtype=np.uint8)}
    assert (
        convert_bair_sequence_to_clip(
            example,
            sample_id="short",
            split="train",
            total_frames=8,
            image_size=224,
        )
        is None
    )
