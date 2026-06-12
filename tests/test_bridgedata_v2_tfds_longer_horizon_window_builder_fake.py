import json
from pathlib import Path

from data.bridgedata_v2_tfds_longer_horizon_manifest import read_jsonl
from data.bridgedata_v2_tfds_longer_horizon_window_builder import (
    build_longer_horizon_windows_from_manifest,
    build_window_from_gap0_record,
)


def _base_record(traj: str, start: int) -> dict:
    return {
        "sample_id": f"{traj}_window_{start:06d}",
        "trajectory_id": traj,
        "split": "train",
        "context_frame_indices": list(range(start, start + 16)),
        "current_frame_indices": list(range(start + 16, start + 20)),
        "future_frame_indices": list(range(start + 20, start + 24)),
        "metadata": {
            "record_metadata": {"tfds_episode_index": int(traj.rsplit("_", 1)[-1])},
            "use_action_as_input": False,
            "use_language_as_input": False,
            "use_goal_image_as_input": False,
        },
    }


def test_build_window_adds_gap_and_moves_future_target():
    record = _base_record("tfds_episode_000001", 3)
    out = build_window_from_gap0_record(record, horizon_gap=8)
    assert out["context_frame_indices"] == list(range(3, 19))
    assert out["current_frame_indices"] == list(range(19, 23))
    assert out["gap_frame_indices"] == list(range(23, 31))
    assert out["future_frame_indices"] == list(range(31, 35))
    assert out["total_span"] == 32
    assert out["metadata"]["use_action_as_input"] is False


def test_build_longer_horizon_manifest_selects_required_gaps_and_safe_skips_short(tmp_path: Path):
    base = [_base_record(f"tfds_episode_{traj:06d}", start) for traj in range(4) for start in range(24)]
    manifest = tmp_path / "base.jsonl"
    manifest.write_text("\n".join(json.dumps(item) for item in base) + "\n", encoding="utf-8")
    out_manifest = tmp_path / "horizon.jsonl"
    out_summary = tmp_path / "summary.json"
    summary = build_longer_horizon_windows_from_manifest(
        manifest,
        out_manifest,
        out_summary,
        horizon_gaps=[0, 4, 8, 12],
        required_horizon_gaps=[0, 4, 8],
        optional_horizon_gaps=[12],
        target_windows_per_horizon=16,
        min_windows_per_horizon=8,
        max_windows_per_trajectory_soft_cap=8,
    )
    assert summary["safe_stop"] is False
    assert summary["horizons"]["gap0"]["selected_windows"] == 16
    assert summary["horizons"]["gap4"]["selected_windows"] == 16
    assert summary["horizons"]["gap8"]["selected_windows"] == 16
    assert read_jsonl(out_manifest)

