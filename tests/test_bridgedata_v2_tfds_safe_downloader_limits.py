from pathlib import Path

from data.bridgedata_v2_tfds_safe_downloader import (
    download_tfds_mini_shards,
    is_forbidden_raw_url,
    validate_download_item,
)


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        if self.offset >= len(self.payload):
            return b""
        if size < 0:
            size = len(self.payload) - self.offset
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


def _config(tmp_path: Path, max_bytes: int = 100):
    return {
        "download_policy": {
            "max_download_bytes": max_bytes,
            "allow_metadata_download": True,
        },
        "paths": {
            "download_root": str(tmp_path / "download"),
            "tfds_dataset_root": str(tmp_path / "download" / "bridge_dataset" / "1.0.0"),
        },
    }


def test_unknown_and_large_content_length_are_rejected():
    assert validate_download_item("https://example.test/shard", None, max_download_bytes=100)[0] is False
    assert validate_download_item("https://example.test/shard", 101, max_download_bytes=100)[0] is False
    assert is_forbidden_raw_url("https://example.test/demos_8_17.zip") is True


def test_safe_downloader_rejects_unknown_train_shard_size(tmp_path):
    inventory = {
        "safe_stop": False,
        "selected_under_limit": True,
        "metadata_files": [],
        "selected_shards": [{"name": "bridge_dataset-train.tfrecord-00000-of-00001", "url": "https://example.test/shard"}],
    }
    summary = download_tfds_mini_shards(_config(tmp_path), inventory, head_func=lambda url: None)
    assert summary["safe_stop"] is True
    assert summary["download_performed"] is False


def test_safe_downloader_deletes_partial_when_stream_exceeds_limit(tmp_path):
    inventory = {
        "safe_stop": False,
        "selected_under_limit": True,
        "metadata_files": [],
        "selected_shards": [{"name": "bridge_dataset-train.tfrecord-00000-of-00001", "url": "https://example.test/shard"}],
    }
    summary = download_tfds_mini_shards(
        _config(tmp_path, max_bytes=5),
        inventory,
        head_func=lambda url: 5,
        open_func=lambda url: FakeResponse(b"123456"),
    )
    assert summary["safe_stop"] is True
    assert not list(tmp_path.glob("**/*.partial"))
    assert not list(tmp_path.glob("**/*.tfrecord-*"))
