# BridgeData V2 TFDS Local Usage

Step23 stores local mini-shard data under an ignored directory:

```text
data/bridgedata_v2_tfds_mini/
  bridge_dataset/
    1.0.0/
      dataset_info.json
      features.json
      bridge_dataset-train.tfrecord-00000-of-01024
```

The exact shard names depend on the official TFDS listing. Runtime outputs are written
under:

```text
runs/bridgedata_v2_tfds_mini_shard_step23_v1/
```

Both locations are ignored by git.

## Keep `env_isaaclab` Clean

Run project tests and smoke orchestration with `env_isaaclab`, but do not install
TensorFlow or TensorFlow Datasets there.

```bash
/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_bridgedata_v2_tfds_mini_shard.py
```

The smoke script calls the isolated TFDS environment only for RLDS inspection:

```text
/home/ubuntu22/miniconda3/envs/tgpawb_tfds_py311/bin/python
```

## Expanding Shard Count

The config defaults to one preferred train shard and allows at most five, with total
download bytes capped at 1GB. Increase shard count only by editing:

```yaml
download_policy:
  preferred_tfds_train_shards: 1
  max_tfds_train_shards: 5
  max_download_bytes: 1073741824
```

Every shard must have a known `Content-Length` before download.

## When Cloud Is Needed

This local smoke is enough to validate schema and window construction. Use a cloud
machine only for larger TFDS coverage, full BridgeData V2 processing, or later
compute-heavy token extraction.
