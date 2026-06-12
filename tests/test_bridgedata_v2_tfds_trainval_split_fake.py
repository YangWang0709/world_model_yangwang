from data.bridgedata_v2_tfds_trainval_split import build_train_val_split
from tests.test_bridgedata_v2_tfds_window_sampler_fake import _fake_windows
from data.bridgedata_v2_tfds_window_sampler import select_diverse_windows


def test_trainval_split_is_deterministic_disjoint_and_near_24_8():
    selected, _ = select_diverse_windows(_fake_windows(), target_num_windows=32)
    split_a = build_train_val_split(selected, train_ratio=0.75, min_train_windows=24, min_val_windows=8)
    split_b = build_train_val_split(selected, train_ratio=0.75, min_train_windows=24, min_val_windows=8)
    assert split_a == split_b
    assert split_a["num_train_windows"] == 24
    assert split_a["num_val_windows"] == 8
    assert split_a["train_val_trajectory_disjoint"] is True
    assert split_a["duplicate_sample_ids_between_train_val"] is False
    assert set(split_a["train_sample_ids"]).isdisjoint(split_a["val_sample_ids"])

