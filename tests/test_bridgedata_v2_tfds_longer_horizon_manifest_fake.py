import pytest

from data.bridgedata_v2_tfds_longer_horizon_manifest import (
    summarize_horizon_manifest,
    validate_longer_horizon_window,
)


def _window(gap: int = 4) -> dict:
    return {
        "sample_id": f"gap{gap}_sample",
        "trajectory_id": "traj0",
        "horizon_gap": gap,
        "total_span": 16 + 4 + gap + 4,
        "context_frame_indices": list(range(0, 16)),
        "current_frame_indices": list(range(16, 20)),
        "gap_frame_indices": list(range(20, 20 + gap)),
        "future_frame_indices": list(range(20 + gap, 24 + gap)),
        "metadata": {
            "use_action_as_input": False,
            "use_language_as_input": False,
            "use_goal_image_as_input": False,
        },
    }


def test_longer_horizon_manifest_schema_and_metadata_only_fields():
    record = _window(8)
    assert validate_longer_horizon_window(record) is True
    summary = summarize_horizon_manifest([record])
    assert summary["horizons"]["gap8"]["selected_windows"] == 1
    assert summary["action_language_goal_as_metadata_only"] is True


def test_gap_frames_must_not_overlap_input_or_target():
    record = _window(4)
    record["future_frame_indices"] = [20, 21, 22, 23]
    with pytest.raises(ValueError):
        validate_longer_horizon_window(record)

