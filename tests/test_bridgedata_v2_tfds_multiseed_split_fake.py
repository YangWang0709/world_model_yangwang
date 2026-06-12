from data.bridgedata_v2_tfds_multiseed_split import build_multiseed_splits


def _fake_windows(num_trajectories=10, windows_per_trajectory=8):
    records = []
    for trajectory in range(num_trajectories):
        for index in range(windows_per_trajectory):
            records.append(
                {
                    "sample_id": f"tfds_episode_{trajectory:06d}_window_{index:06d}",
                    "trajectory_id": f"tfds_episode_{trajectory:06d}",
                    "context_frame_indices": list(range(index, index + 16)),
                    "current_frame_indices": list(range(index + 16, index + 20)),
                    "future_frame_indices": list(range(index + 20, index + 24)),
                }
            )
    return records[:64]


def test_step30a_multiseed_split_is_deterministic_and_disjoint():
    windows = _fake_windows()
    first = build_multiseed_splits(windows, seeds=[42, 123, 999], min_val_windows_by_count={64: 16})
    second = build_multiseed_splits(windows, seeds=[42, 123, 999], min_val_windows_by_count={64: 16})
    assert first == second
    assert first["num_splits"] == 3
    for split in first["splits"]:
        assert split["num_train_windows"] == 48
        assert split["num_val_windows"] == 16
        assert split["train_val_trajectory_disjoint"] is True
        assert split["duplicate_sample_ids_between_train_val"] is False
        assert not (set(split["train_sample_ids"]) & set(split["val_sample_ids"]))


def test_step30a_multiseed_split_reports_fallback_for_single_trajectory():
    windows = _fake_windows(num_trajectories=1, windows_per_trajectory=64)
    splits = build_multiseed_splits(windows, seeds=[42], min_val_windows_by_count={64: 16})
    split = splits["splits"][0]
    assert split["fallback_used"] is True
    assert split["fallback_reason"]
    assert split["train_val_trajectory_disjoint"] is False
    assert split["duplicate_sample_ids_between_train_val"] is False
