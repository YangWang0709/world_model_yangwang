# Token Extraction Smoke Report

Command: `python scripts/smoke_test_token_extraction.py`

- dataset path: `/home/ubuntu22/tgpawb_world_model/data/toy_videos`
- dataset size: `16`
- output shard path: `/home/ubuntu22/tgpawb_world_model/data/token_shards/dummy_toy`
- generated shard count: `2`
- generated shard files: `['tokens_shard_000000.pt', 'tokens_shard_000001.pt']`
- validation result: `True`

## First Shard Summary

```json
{
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
  "split": "toy"
}
```

## Extraction Summary

```json
{
  "dataset_size": 16,
  "batch_size": 4,
  "shard_size": 8,
  "device": "cuda",
  "encoder_name": "dummy_video_encoder",
  "num_shards": 2
}
```

TOKEN_EXTRACTION_SMOKE_PASS = true
