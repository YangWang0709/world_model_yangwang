import json
from pathlib import Path

from data.bridgedata_v2_tfds_longer_horizon_split import build_split_for_horizon, write_longer_horizon_splits


def _record(gap: int, traj: int, idx: int) -> dict:
    return {
        "sample_id": f"gap{gap}_traj{traj}_window_{idx}",
        "trajectory_id": f"traj{traj}",
        "horizon_gap": gap,
    }


def test_build_split_prefers_trajectory_disjoint():
    records = [_record(4, traj, idx) for traj in range(4) for idx in range(6)]
    split = build_split_for_horizon(records, horizon_gap=4, seed=42)
    assert split["num_train_windows"] > 0
    assert split["num_val_windows"] > 0
    assert split["train_val_trajectory_disjoint"] is True
    assert set(split["train_sample_ids"]).isdisjoint(split["val_sample_ids"])
    assert set(split["train_trajectories"]).isdisjoint(split["val_trajectories"])


def test_write_longer_horizon_splits_for_multiple_seeds(tmp_path: Path):
    records = [_record(gap, traj, idx) for gap in (0, 4) for traj in range(3) for idx in range(5)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    payload = write_longer_horizon_splits(manifest, tmp_path / "splits.json", seeds=[42, 123, 999])
    assert payload["seeds"] == [42, 123, 999]
    assert payload["num_splits"] == 6
    for split in payload["splits"]:
        assert set(split["train_sample_ids"]).isdisjoint(split["val_sample_ids"])

