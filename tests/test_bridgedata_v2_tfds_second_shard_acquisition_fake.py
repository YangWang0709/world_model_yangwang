from io import BytesIO
from pathlib import Path

from data.bridgedata_v2_tfds_second_shard_acquisition import acquire_second_shard


class FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_second_shard_reuses_existing_file(tmp_path: Path):
    root = tmp_path / "tfds"
    root.mkdir()
    existing = root / "bridge_dataset-train.tfrecord-00001-of-01024"
    existing.write_bytes(b"abc")
    summary = acquire_second_shard(
        dataset_root=root,
        base_url="https://example.test/",
        preferred_indices=[1],
        expected_filename_pattern="bridge_dataset-train.tfrecord-{index:05d}-of-01024",
        head_func=lambda url: (_ for _ in ()).throw(AssertionError("HEAD should not run")),
    )
    assert summary["safe_stop"] is False
    assert summary["downloaded_new_shard_count"] == 0
    assert summary["reused_existing_shard"] is True


def test_second_shard_downloads_only_one_known_size_file(tmp_path: Path):
    data = b"x" * 12
    summary = acquire_second_shard(
        dataset_root=tmp_path,
        base_url="https://example.test/",
        preferred_indices=[1, 2],
        expected_filename_pattern="bridge_dataset-train.tfrecord-{index:05d}-of-01024",
        hard_cap_size_mb=1,
        head_func=lambda url: len(data),
        open_func=lambda url: FakeResponse(data),
    )
    assert summary["safe_stop"] is False
    assert summary["downloaded_new_shard_count"] == 1
    assert (tmp_path / "bridge_dataset-train.tfrecord-00001-of-01024").read_bytes() == data


def test_second_shard_unknown_or_too_large_safe_stops(tmp_path: Path):
    unknown = acquire_second_shard(
        dataset_root=tmp_path / "unknown",
        base_url="https://example.test/",
        preferred_indices=[1],
        expected_filename_pattern="bridge_dataset-train.tfrecord-{index:05d}-of-01024",
        head_func=lambda url: None,
    )
    assert unknown["safe_stop"] is True
    too_large = acquire_second_shard(
        dataset_root=tmp_path / "large",
        base_url="https://example.test/",
        preferred_indices=[1],
        expected_filename_pattern="bridge_dataset-train.tfrecord-{index:05d}-of-01024",
        hard_cap_size_mb=1,
        head_func=lambda url: 2 * 1024 * 1024,
    )
    assert too_large["safe_stop"] is True


def test_second_shard_rejects_raw_zip_pattern(tmp_path: Path):
    summary = acquire_second_shard(
        dataset_root=tmp_path,
        base_url="https://example.test/",
        preferred_indices=[1],
        expected_filename_pattern="demos_{index:05d}.zip",
        head_func=lambda url: 1,
    )
    assert summary["safe_stop"] is True
    assert summary["downloaded_new_shard_count"] == 0
