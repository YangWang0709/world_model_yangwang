# VideoMAE Real Encoder Smoke Report

Commands:
- `python scripts/smoke_test_videomae_real_encoder.py`
- `python scripts/smoke_test_real_video_videomae_token_extraction.py`

## Encoder Smoke

- model_name: `/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local`
- cache_dir: `/home/ubuntu22/tgpawb_world_model/model_cache/huggingface`
- device: `cuda`
- input_shape: `[1, 8, 3, 224, 224]`
- output_shape: `[1, 784, 768]`
- dtype: `torch.float32`
- GPU memory before: `{'free_gib': 14.139, 'total_gib': 15.461, 'used_gib': 1.321}`
- GPU memory after: `{'free_gib': 14.03, 'total_gib': 15.461, 'used_gib': 1.43}`
- RAM before: `{'total_gib': 30.403, 'available_gib': 25.521, 'used_gib': 4.392}`
- RAM after: `{'total_gib': 30.403, 'available_gib': 25.281, 'used_gib': 4.629}`
- elapsed_time_sec: `0.106`
- oom: `False`
- cloud_recommendation: `not required for Step 9C; local RTX 5080 conservative smoke succeeded`
- pass: `True`

```json
{
  "download_ok": true,
  "encoder_available": true,
  "rank_ok": true,
  "batch_ok": true,
  "finite_ok": true,
  "nonempty_tokens": true,
  "oom_ok": true
}
```

VIDEOMAE_REAL_ENCODER_SMOKE_PASS = true

## Real Video Token Extraction

```json
{
  "output_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke",
  "generated_shard_count": 4,
  "generated_shard_files": [
    "tokens_shard_000000.pt",
    "tokens_shard_000001.pt",
    "tokens_shard_000002.pt",
    "tokens_shard_000003.pt"
  ],
  "first_shard_summary": {
    "schema_version": "0.1.0",
    "encoder_name": "videomae",
    "num_samples": 2,
    "past_tokens_shape": [
      2,
      784,
      768
    ],
    "future_tokens_shape": [
      2,
      784,
      768
    ],
    "future_tokens_rank": 3,
    "split": "real_minimal"
  },
  "first_shard_encoder_config": {
    "requested_encoder": "videomae",
    "actual_encoder": "videomae",
    "used_fallback": false,
    "fallback_reason": null,
    "model_name_or_path": "/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local",
    "cache_dir": "/home/ubuntu22/tgpawb_world_model/model_cache/huggingface",
    "output_mode": "last_hidden_state",
    "image_size": 224,
    "num_frames": 8,
    "token_dim": 768,
    "num_tokens": 784,
    "dummy_num_tokens": 196,
    "dummy_token_dim": 768,
    "patch_grid_h": 14,
    "patch_grid_w": 14,
    "encoder_availability": {
      "available": true,
      "reason": "VideoMAE model loaded",
      "model_name_or_path": "/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local",
      "cache_dir": "/home/ubuntu22/tgpawb_world_model/model_cache/huggingface",
      "allow_download": false,
      "local_files_only": true,
      "device": "cuda",
      "pretrained_num_frames": 16,
      "effective_num_frames": 8,
      "image_size": 224,
      "output_mode": "last_hidden_state",
      "ignore_mismatched_sizes": true
    }
  },
  "requested_encoder": "videomae",
  "actual_encoder": "videomae",
  "used_fallback": false,
  "model_name_or_path": "/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local",
  "cache_dir": "/home/ubuntu22/tgpawb_world_model/model_cache/huggingface",
  "extraction_summary_path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke/extraction_summary.json",
  "extraction_summary": {
    "dataset_size": 8,
    "batch_size": 1,
    "max_samples": 8,
    "shard_size": 2,
    "device": "cuda",
    "encoder_name": "videomae",
    "num_shards": 4,
    "elapsed_time_sec": 0.781
  },
  "resource": {
    "ram_before": {
      "total_gib": 30.403,
      "available_gib": 25.242,
      "used_gib": 4.668
    },
    "ram_after": {
      "total_gib": 30.403,
      "available_gib": 25.149,
      "used_gib": 4.761
    },
    "gpu_memory_before": {
      "free_gib": 14.03,
      "total_gib": 15.461,
      "used_gib": 1.43
    },
    "gpu_memory_after": {
      "free_gib": 14.03,
      "total_gib": 15.461,
      "used_gib": 1.43
    },
    "elapsed_time_sec": 0.782,
    "oom": false
  },
  "checks": {
    "encoder_smoke_passed": true,
    "shard_exists": true,
    "past_tokens_finite": true,
    "future_tokens_finite": true,
    "split_ok": true,
    "encoder_name_ok": true,
    "summary_exists": true,
    "requested_encoder_ok": true,
    "actual_encoder_ok": true,
    "used_fallback_false": true,
    "oom_ok": true
  },
  "pass": true,
  "error": null
}
```

REAL_VIDEO_VIDEOMAE_TOKEN_EXTRACTION_SMOKE_PASS = true
