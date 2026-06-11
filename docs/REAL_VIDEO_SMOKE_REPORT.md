# Real Video Smoke Report

Commands:
- `python scripts/smoke_test_real_video_dataset.py`
- `python scripts/smoke_test_real_video_token_extraction.py`

## Resource Summary

```json
{
  "cpu_count": 32,
  "ram_total_gib": 30.403,
  "ram_available_gib": 25.484,
  "disk_free_gib": 273.11,
  "disk_total_gib": 483.587,
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 5080",
  "gpu_memory_total_gib": 15.461,
  "gpu_memory_free_gib": 14.378,
  "recommended": {
    "batch_size": 2,
    "num_workers": 0,
    "max_samples": 100,
    "clip_len": 8,
    "image_size": 224
  },
  "warnings": []
}
```

## Dataset Smoke

- dataset path: `/home/ubuntu22/tgpawb_world_model/data/real_video_minimal`
- metadata path: `/home/ubuntu22/tgpawb_world_model/data/real_video_minimal/metadata.jsonl`
- number of samples: `16`
- first sample id: `real_000000`
- first past shape: `[4, 3, 224, 224]`
- first future shape: `[4, 3, 224, 224]`
- value range: `[0.15905803442001343, 0.6696805953979492]`

```json
{
  "num_samples_ok": true,
  "past_shape_ok": true,
  "future_shape_ok": true,
  "finite_ok": true,
  "value_range_ok": true
}
```

REAL_VIDEO_DATASET_SMOKE_PASS = true

## Token Extraction Smoke

```json
{
  "output_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_dummy",
  "generated_shard_count": 2,
  "generated_shard_files": [
    "tokens_shard_000000.pt",
    "tokens_shard_000001.pt"
  ],
  "first_shard_summary": {
    "schema_version": "0.1.0",
    "encoder_name": "dummy_video_encoder",
    "num_samples": 8,
    "past_tokens_shape": [
      8,
      196,
      768
    ],
    "future_tokens_shape": [
      8,
      196,
      768
    ],
    "future_tokens_rank": 3,
    "split": "real_minimal"
  },
  "extraction_summary": {
    "dataset_size": 16,
    "batch_size": 2,
    "shard_size": 8,
    "device": "cuda",
    "encoder_name": "dummy_video_encoder",
    "num_shards": 2
  },
  "checks": {
    "shard_count_ok": true,
    "past_tokens_shape_ok": true,
    "future_tokens_shape_ok": true,
    "split_ok": true,
    "metadata_source_type_ok": true,
    "metadata_path_ok": true
  },
  "pass": true
}
```

REAL_VIDEO_TOKEN_EXTRACTION_SMOKE_PASS = true
