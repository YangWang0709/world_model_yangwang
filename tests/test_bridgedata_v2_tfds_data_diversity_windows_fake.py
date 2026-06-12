from data.bridgedata_v2_tfds_data_diversity_windows import build_fake_gap0_windows
from data.bridgedata_v2_tfds_longer_horizon_manifest import validate_longer_horizon_window


def test_fake_shard_windows_have_gap0_indices_and_metadata_only():
    windows, summary = build_fake_gap0_windows({"traj_a": 40, "traj_b": 40}, target_windows=8, min_windows=4)
    assert summary["safe_stop"] is False
    assert len(windows) == 8
    for window in windows:
        assert validate_longer_horizon_window(window)
        assert window["horizon_gap"] == 0
        assert window["context_frame_indices"] == list(range(window["context_frame_indices"][0], window["context_frame_indices"][0] + 16))
        assert window["metadata"]["use_action_as_input"] is False
        assert window["metadata"]["use_language_as_input"] is False


def test_fake_shard_windows_safe_stop_when_insufficient():
    windows, summary = build_fake_gap0_windows({"traj_a": 24}, target_windows=64, min_windows=32)
    assert windows == []
    assert summary["safe_stop"] is True
    assert summary["available_windows"] == 1
