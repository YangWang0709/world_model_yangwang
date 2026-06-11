# Frozen Video Encoder Smoke Report

Commands:
- `python scripts/smoke_test_frozen_video_encoder.py`
- `python scripts/smoke_test_real_video_frozen_token_extraction.py`

## Frozen Encoder Smoke

- requested encoder: `videomae`
- actual encoder: `dummy_video_encoder`
- used fallback: `True`
- fallback reason: `model_name_or_path is not configured`
- output token shape: `[1, 196, 768]`
- GPU memory before: `{'free_gib': 14.438, 'total_gib': 15.461}`
- GPU memory after: `{'free_gib': 14.378, 'total_gib': 15.461}`

```json
{
  "rank_ok": true,
  "batch_ok": true,
  "finite_ok": true
}
```

FROZEN_VIDEO_ENCODER_SMOKE_PASS = true

## Real Video Frozen Token Extraction Smoke

```json
{
  "output_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_frozen_encoder_smoke",
  "generated_shard_count": 2,
  "generated_shard_files": [
    "tokens_shard_000000.pt",
    "tokens_shard_000001.pt"
  ],
  "first_shard_summary": {
    "schema_version": "0.1.0",
    "encoder_name": "dummy_video_encoder",
    "num_samples": 2,
    "past_tokens_shape": [
      2,
      196,
      768
    ],
    "future_tokens_shape": [
      2,
      196,
      768
    ],
    "future_tokens_rank": 3,
    "split": "real_minimal"
  },
  "requested_encoder": "videomae",
  "actual_encoder": "dummy_video_encoder",
  "used_fallback": true,
  "fallback_reason": "model_name_or_path is not configured",
  "extraction_summary": {
    "dataset_size": 4,
    "batch_size": 1,
    "shard_size": 2,
    "device": "cuda",
    "encoder_name": "dummy_video_encoder",
    "num_shards": 2
  },
  "checks": {
    "shard_exists": true,
    "past_tokens_finite": true,
    "future_tokens_finite": true,
    "split_ok": true,
    "encoder_name_ok": true,
    "summary_exists": true
  },
  "pass": true
}
```

REAL_VIDEO_FROZEN_TOKEN_EXTRACTION_SMOKE_PASS = true
