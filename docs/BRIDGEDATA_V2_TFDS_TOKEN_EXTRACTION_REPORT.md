# BridgeData V2 TFDS Token Extraction Report

- pass: `true`
- safe_stop: `false`
- model_missing: `false`
- image_field: `steps/observation/image_0`
- image_field_is_metadata_flag: `false`
- clip_export_performed: `true`
- token_extraction_performed: `true`
- num_windows_exported: `4`
- num_windows_tokenized: `4`
- context_clip_shape_example: `[16, 3, 224, 224]`
- current_clip_shape_example: `[4, 3, 224, 224]`
- future_clip_shape_example: `[4, 3, 224, 224]`
- context_token_shape_example: `[16, 392, 768]`
- current_token_shape_example: `[4, 392, 768]`
- future_token_shape_example: `[4, 392, 768]`
- model_loaded_local_only: `true`
- model_download_performed: `false`
- training_performed: `false`
- importance_generation_performed: `false`
- large_token_shards_generated: `false`
- data_token_shards_written: `false`
- safety_gate_pass: `true`
- recommended_step25: `BridgeData V2 TFDS mini-shard predictive importance dry-run`

## Clip Export Summary

```json
{
  "clip_cache_dir": "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_token_extraction_step24_v1/clip_cache",
  "clip_cache_files": [
    "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_token_extraction_step24_v1/clip_cache/tfds_episode_000000_window_000000.npz",
    "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_token_extraction_step24_v1/clip_cache/tfds_episode_000000_window_000001.npz",
    "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_token_extraction_step24_v1/clip_cache/tfds_episode_000000_window_000002.npz",
    "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_token_extraction_step24_v1/clip_cache/tfds_episode_000000_window_000003.npz"
  ],
  "clip_cache_total_bytes": 13762813,
  "clip_export_performed": true,
  "context_shape": [
    16,
    3,
    224,
    224
  ],
  "current_shape": [
    4,
    3,
    224,
    224
  ],
  "download_performed": false,
  "future_shape": [
    4,
    3,
    224,
    224
  ],
  "image_field": "steps/observation/image_0",
  "image_field_is_metadata_flag": false,
  "importance_generation_performed": false,
  "num_windows_exported": 4,
  "num_windows_requested": 4,
  "reason": null,
  "safe_stop": false,
  "safety_gate_pass": true,
  "stage": "bridgedata_v2_tfds_clip_export",
  "token_extraction_performed": false,
  "training_performed": false
}
```

## Token Summary

```json
{
  "clip_export_summary": {
    "clip_cache_dir": "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_token_extraction_step24_v1/clip_cache",
    "clip_cache_files": [
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_token_extraction_step24_v1/clip_cache/tfds_episode_000000_window_000000.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_token_extraction_step24_v1/clip_cache/tfds_episode_000000_window_000001.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_token_extraction_step24_v1/clip_cache/tfds_episode_000000_window_000002.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_token_extraction_step24_v1/clip_cache/tfds_episode_000000_window_000003.npz"
    ],
    "clip_cache_total_bytes": 13762813,
    "clip_export_performed": true,
    "context_shape": [
      16,
      3,
      224,
      224
    ],
    "current_shape": [
      4,
      3,
      224,
      224
    ],
    "download_performed": false,
    "future_shape": [
      4,
      3,
      224,
      224
    ],
    "image_field": "steps/observation/image_0",
    "image_field_is_metadata_flag": false,
    "importance_generation_performed": false,
    "num_windows_exported": 4,
    "num_windows_requested": 4,
    "reason": null,
    "safe_stop": false,
    "safety_gate_pass": true,
    "stage": "bridgedata_v2_tfds_clip_export",
    "token_extraction_performed": false,
    "training_performed": false
  },
  "context_token_shape_example": [
    16,
    392,
    768
  ],
  "current_token_shape_example": [
    4,
    392,
    768
  ],
  "data_token_shards_written": false,
  "encoder_availability": {
    "allow_download": false,
    "available": true,
    "cache_dir": null,
    "device": "cuda",
    "effective_num_frames": 4,
    "ignore_mismatched_sizes": true,
    "image_size": 224,
    "local_files_only": true,
    "model_name_or_path": "/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local",
    "output_mode": "last_hidden_state",
    "pretrained_num_frames": 16,
    "reason": "VideoMAE model loaded"
  },
  "env_isaaclab_guard": {
    "tensorflow": false,
    "tensorflow_datasets": false
  },
  "future_token_shape_example": [
    4,
    392,
    768
  ],
  "image_field": "steps/observation/image_0",
  "image_field_is_metadata_flag": false,
  "importance_generation_performed": false,
  "large_token_shards_generated": false,
  "model_download_performed": false,
  "model_loaded_local_only": true,
  "model_missing": false,
  "model_path": "/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local",
  "num_token_artifacts": 4,
  "num_windows_tokenized": 4,
  "reason": null,
  "safe_stop": false,
  "safety_gate_pass": true,
  "stage": "bridgedata_v2_tfds_token_extraction_step24",
  "token_extraction_performed": true,
  "token_manifest_jsonl": "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_token_extraction_step24_v1/tfds_token_smoke_manifest.jsonl",
  "token_smoke_dir": "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_token_extraction_step24_v1/token_smoke",
  "training_performed": false
}
```
