from pathlib import Path

from data.bridgedata_v2_manifest_schema import make_fake_bridgedata_manifest_records
from data.bridgedata_v2_window_builder import (
    build_bridgedata_windows_from_manifest,
    summarize_window_records,
    write_window_manifest_jsonl,
)
from data.long_context_dataset_schema import validate_long_context_sample
from data.long_context_window_spec import LongContextWindowSpec


def test_fake_manifest_builds_long_context_windows(tmp_path):
    records = make_fake_bridgedata_manifest_records(num_trajectories=3, num_frames=40)
    spec = LongContextWindowSpec(context_len=16, current_len=4, future_len=4)
    windows = build_bridgedata_windows_from_manifest(records, spec)
    assert windows
    assert len(windows) == 3 * 17
    for window in windows[:5]:
        assert validate_long_context_sample(window)
        assert window["context_video"] is None
        assert window["metadata"]["use_action_as_input"] is False
        assert window["metadata"]["use_language_as_input"] is False
        assert window["metadata"]["use_goal_image_as_input"] is False
        assert len(window["context_frame_indices"]) == 16
        assert len(window["current_frame_indices"]) == 4
        assert len(window["future_frame_indices"]) == 4
    summary = summarize_window_records(windows)
    assert summary["num_windows"] == len(windows)
    assert summary["action_used_as_input"] is False
    assert summary["language_used_as_input"] is False
    assert summary["goal_image_used_as_input"] is False
    out = write_window_manifest_jsonl(tmp_path / "windows.jsonl", windows)
    assert out.exists()
    assert not list(Path(tmp_path).glob("*.pt"))


def test_short_trajectory_is_skipped():
    records = make_fake_bridgedata_manifest_records(num_trajectories=1, num_frames=23)
    spec = LongContextWindowSpec(context_len=16, current_len=4, future_len=4)
    assert build_bridgedata_windows_from_manifest(records, spec) == []
