import pytest

from data.bridgedata_v2_manifest_schema import (
    make_fake_bridgedata_manifest_records,
    summarize_bridgedata_manifest_record,
    validate_bridgedata_manifest_record,
)


def test_fake_manifest_record_validates_and_summarizes():
    record = make_fake_bridgedata_manifest_records(num_trajectories=1, num_frames=40)[0]
    assert validate_bridgedata_manifest_record(record)
    summary = summarize_bridgedata_manifest_record(record)
    assert summary["dataset_name"] == "BridgeData V2"
    assert summary["num_frames"] == 40
    assert summary["has_frame_paths"] is True
    assert summary["has_language_instruction"] is True


def test_manifest_record_requires_trajectory_id():
    with pytest.raises(ValueError, match="trajectory_id"):
        validate_bridgedata_manifest_record({"num_frames": 40})


def test_manifest_record_rejects_invalid_num_frames():
    with pytest.raises(ValueError, match="num_frames"):
        validate_bridgedata_manifest_record({"trajectory_id": "traj0", "num_frames": 0})


def test_manifest_record_strict_frame_path_count_check():
    record = {
        "trajectory_id": "traj0",
        "num_frames": 3,
        "frame_paths": ["a.jpg", "b.jpg"],
    }
    assert validate_bridgedata_manifest_record(record, strict=False)
    with pytest.raises(ValueError, match="len\\(frame_paths\\)"):
        validate_bridgedata_manifest_record(record, strict=True)


def test_manifest_optional_metadata_fields_can_be_missing():
    record = {"trajectory_id": "traj0", "num_frames": 24}
    assert validate_bridgedata_manifest_record(record)
    summary = summarize_bridgedata_manifest_record(record)
    assert summary["has_actions_path"] is False
    assert summary["has_language_instruction"] is False
    assert summary["has_goal_image_path"] is False
    assert summary["camera_count"] == 0
