from data.bridgedata_v2_tfds_inventory import parse_size_to_bytes, parse_tfds_index_html, select_train_shards


FAKE_HTML = """
<html><body><pre>
<a href="dataset_info.json">dataset_info.json</a>        2023-09-21 12:42  9.5K
<a href="features.json">features.json</a>            2023-09-21 12:42  4.0K
<a href="action_proprio_stats.json">action_proprio_stats.json</a> 2023-09-21 12:42 1.2K
<a href="bridge_dataset-train.tfrecord-00000-of-01024">bridge_dataset-train.tfrecord-00000-of-01024</a> 2023-09-21 12:42 90M
<a href="bridge_dataset-train.tfrecord-00001-of-01024">bridge_dataset-train.tfrecord-00001-of-01024</a> 2023-09-21 12:42 103M
<a href="bridge_dataset-train.tfrecord-00002-of-01024">bridge_dataset-train.tfrecord-00002-of-01024</a> 2023-09-21 12:42 131M
</pre></body></html>
"""


def test_size_parser_handles_apache_units():
    assert parse_size_to_bytes("90M") == 90 * 1024 * 1024
    assert parse_size_to_bytes("103M") == 103 * 1024 * 1024
    assert parse_size_to_bytes("131M") == 131 * 1024 * 1024


def test_inventory_parser_finds_metadata_and_train_shards():
    parsed = parse_tfds_index_html(
        FAKE_HTML,
        "https://example.test/tfds/bridge_dataset/1.0.0/",
        metadata_files=["dataset_info.json", "features.json"],
        optional_metadata_patterns=["action_proprio_stats*.json"],
        shard_name_prefix="bridge_dataset-train",
    )
    assert {item["name"] for item in parsed["metadata_files"]} == {
        "dataset_info.json",
        "features.json",
        "action_proprio_stats.json",
    }
    assert len(parsed["train_shards"]) == 3
    selected, total, under = select_train_shards(
        parsed["train_shards"],
        preferred_shard_count=2,
        max_shard_count=5,
        max_download_bytes=220 * 1024 * 1024,
    )
    assert [item["name"] for item in selected] == [
        "bridge_dataset-train.tfrecord-00000-of-01024",
        "bridge_dataset-train.tfrecord-00001-of-01024",
    ]
    assert total <= 220 * 1024 * 1024
    assert under is True
