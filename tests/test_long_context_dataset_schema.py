import pytest
import torch

from data.long_context_dataset_schema import (
    make_placeholder_long_context_sample,
    summarize_long_context_sample,
    validate_long_context_sample,
)


def test_fake_long_context_sample_validates_and_summarizes():
    sample = make_placeholder_long_context_sample(
        dataset_name="BridgeData V2",
        trajectory_id="traj0",
        window={
            "context_frame_indices": list(range(16)),
            "current_frame_indices": list(range(16, 20)),
            "future_frame_indices": list(range(20, 24)),
        },
    )
    assert validate_long_context_sample(sample)
    summary = summarize_long_context_sample(sample)
    assert summary["context_len"] == 16
    assert summary["current_len"] == 4
    assert summary["future_len"] == 4


def test_optional_action_language_goal_fields_can_be_none_or_missing():
    sample = {
        "dataset_name": "DROID",
        "split": "dryrun",
        "trajectory_id": "episode0",
        "sample_id": "episode0_start_0",
        "context_frame_indices": list(range(8)),
        "current_frame_indices": list(range(8, 12)),
        "future_frame_indices": list(range(12, 16)),
        "metadata": {},
    }
    assert validate_long_context_sample(sample)
    summary = summarize_long_context_sample(sample)
    assert summary["has_actions"] is False
    assert summary["has_language_instruction"] is False
    assert summary["has_goal_image"] is False


def test_strict_mode_rejects_bad_video_shape():
    sample = make_placeholder_long_context_sample(
        dataset_name="BAIR",
        trajectory_id="traj0",
        window={
            "context_frame_indices": list(range(8)),
            "current_frame_indices": list(range(8, 12)),
            "future_frame_indices": list(range(12, 16)),
        },
    )
    sample["context_video"] = torch.zeros(8, 1, 64, 64)
    with pytest.raises(ValueError, match="channel dimension"):
        validate_long_context_sample(sample, strict=True)


def test_strict_mode_accepts_fake_video_tensors():
    sample = make_placeholder_long_context_sample(
        dataset_name="BAIR",
        trajectory_id="traj0",
        window={
            "context_frame_indices": list(range(8)),
            "current_frame_indices": list(range(8, 12)),
            "future_frame_indices": list(range(12, 16)),
        },
    )
    sample["context_video"] = torch.zeros(8, 3, 64, 64)
    sample["current_video"] = torch.zeros(4, 3, 64, 64)
    sample["future_video"] = torch.zeros(4, 3, 64, 64)
    assert validate_long_context_sample(sample, strict=True)
