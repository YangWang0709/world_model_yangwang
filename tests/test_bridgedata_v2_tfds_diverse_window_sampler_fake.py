from data.bridgedata_v2_tfds_diverse_window_sampler import select_trajectory_diverse_windows


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


def test_step30a_sampler_selects_64_unique_diverse_windows_without_download():
    selected, summary = select_trajectory_diverse_windows(_fake_windows(), target_count=64)
    assert len(selected) == 64
    assert len({record["sample_id"] for record in selected}) == 64
    assert summary["num_available_windows"] == 155
    assert summary["num_selected_windows"] == 64
    assert summary["num_selected_trajectories"] == 10
    assert summary["new_tfds_shard_downloaded"] is False
    assert summary["download_performed"] is False


def test_step30a_sampler_balances_per_trajectory_counts():
    selected, summary = select_trajectory_diverse_windows(_fake_windows(), target_count=64)
    counts = list(summary["trajectory_window_counts"].values())
    assert max(counts) - min(counts) <= 1
    assert all(record["step30a_selected_rank"] < 64 for record in selected)


def test_step30a_sampler_hard_caps_128_request():
    selected, summary = select_trajectory_diverse_windows(_fake_windows(), target_count=155, hard_cap_windows=128)
    assert len(selected) == 128
    assert summary["num_selected_windows"] == 128
    assert summary["fallback_used"] is True
