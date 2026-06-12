from data.bridgedata_v2_tfds_data_diversity_splits import build_shard_split


def _records():
    return [
        {"sample_id": f"a_{i}", "trajectory_id": "a"} for i in range(4)
    ] + [
        {"sample_id": f"b_{i}", "trajectory_id": "b"} for i in range(4)
    ]


def test_shard_split_is_seeded_and_disjoint():
    split = build_shard_split(_records(), split_seed=42, val_ratio=0.25)
    assert split["split_seed"] == 42
    assert set(split["train_sample_ids"]).isdisjoint(set(split["val_sample_ids"]))
    assert split["trajectory_disjoint"] is True


def test_shard_split_falls_back_with_reason_for_single_trajectory():
    split = build_shard_split([{"sample_id": f"a_{i}", "trajectory_id": "a"} for i in range(8)], split_seed=123)
    assert split["fallback_used"] is True
    assert split["fallback_reason"]
    assert set(split["train_sample_ids"]).isdisjoint(set(split["val_sample_ids"]))
