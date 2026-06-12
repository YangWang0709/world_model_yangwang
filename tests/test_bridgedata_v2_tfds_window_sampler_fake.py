from data.bridgedata_v2_tfds_window_sampler import select_diverse_windows


def _fake_windows(num_trajectories=10, windows_per_trajectory=16):
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
    return records[:155]


def test_window_sampler_selects_32_unique_diverse_windows_without_download():
    selected, summary = select_diverse_windows(_fake_windows(), target_num_windows=32)
    assert len(selected) == 32
    assert len({record["sample_id"] for record in selected}) == 32
    assert summary["num_available_windows"] == 155
    assert summary["num_selected_windows"] == 32
    assert summary["num_selected_trajectories"] == 10
    assert summary["download_performed"] is False


def test_window_sampler_hard_caps_optional_64():
    selected, summary = select_diverse_windows(_fake_windows(), target_num_windows=96, max_windows_hard_cap=64)
    assert len(selected) == 64
    assert summary["num_selected_windows"] == 64

