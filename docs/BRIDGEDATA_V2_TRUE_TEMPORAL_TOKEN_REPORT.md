# BridgeData V2 True-Temporal Token Report

- tokenization_mode: `true_temporal_clip`
- limited_true_temporal_token_extraction_performed: `true`
- num_samples: `64`
- raw_token_shapes: `{'context_example': [1, 1568, 768], 'current_example': [1, 1568, 768], 'future_example': [1, 1568, 768]}`
- token_shapes: `{'context': [1568, 768], 'current': [1568, 768], 'future': [1568, 768]}`
- summary_shapes: `{'context_summary': [768], 'current_summary': [768], 'future_summary': [768]}`
- temporal_spatial_summary_available: `true`
- model_download_performed: `false`
- videomae_training_performed: `false`
- data_token_shards_written: `false`

## Token Summary JSON

```json
{
  "clip_cache_reused_from_step32": true,
  "clip_export_summary": {
    "clip_cache_dir": "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache",
    "clip_cache_files": [
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000000_window_000000.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000001_window_000000.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000002_window_000000.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000003_window_000000.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000004_window_000000.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000005_window_000000.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000006_window_000000.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000007_window_000000.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000008_window_000000.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000009_window_000000.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000000_window_000001.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000001_window_000001.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000002_window_000001.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000003_window_000001.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000004_window_000001.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000005_window_000001.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000006_window_000001.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000007_window_000001.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000008_window_000001.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000009_window_000001.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000000_window_000002.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000001_window_000002.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000002_window_000002.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000003_window_000002.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000004_window_000002.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000005_window_000002.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000006_window_000002.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000007_window_000002.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000008_window_000002.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000000_window_000003.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000001_window_000003.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000002_window_000003.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000003_window_000003.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000004_window_000003.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000005_window_000003.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000006_window_000003.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000007_window_000003.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000008_window_000003.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000000_window_000004.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000001_window_000004.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000002_window_000004.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000003_window_000004.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000004_window_000004.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000006_window_000004.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000007_window_000004.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000008_window_000004.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000001_window_000005.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000002_window_000005.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000003_window_000005.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000004_window_000005.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000006_window_000005.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000007_window_000005.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000008_window_000005.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000001_window_000006.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000002_window_000006.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000003_window_000006.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000004_window_000006.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000006_window_000006.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000007_window_000006.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000008_window_000006.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000001_window_000007.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000002_window_000007.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000003_window_000007.npz",
      "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_longer_horizon_step32_v1/clip_cache/gap0_tfds_episode_000004_window_000007.npz"
    ],
    "clip_cache_reused_from_step32": true,
    "clip_cache_total_bytes": 198895170,
    "clip_export_performed": true,
    "download_performed": false,
    "importance_generation_performed": false,
    "limited_clip_export_performed": false,
    "model_download_performed": false,
    "new_tfds_shard_downloaded": false,
    "num_windows_exported": 64,
    "num_windows_requested": 64,
    "reason": null,
    "safe_stop": false,
    "safety_gate_pass": true,
    "stage": "bridgedata_v2_tfds_true_temporal_gap0_clip_cache",
    "token_extraction_performed": false,
    "training_performed": false
  },
  "context_token_shape_example": [
    1568,
    768
  ],
  "current_token_shape_example": [
    1568,
    768
  ],
  "data_token_shards_written": false,
  "encoder_availability": {
    "allow_download": false,
    "available": true,
    "cache_dir": null,
    "device": "cuda",
    "effective_num_frames": 16,
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
    1568,
    768
  ],
  "horizon_gap": 0,
  "large_token_shards_generated": false,
  "last_encode_summary": {
    "device": "cuda",
    "effective_input_shape": [
      1,
      16,
      3,
      224,
      224
    ],
    "input_shape": [
      1,
      16,
      3,
      224,
      224
    ],
    "output_mode": "last_hidden_state",
    "output_shape": [
      1,
      1568,
      768
    ]
  },
  "limited_clip_export_performed": false,
  "limited_true_temporal_token_extraction_performed": true,
  "model_download_performed": false,
  "model_loaded_local_only": true,
  "model_path": "/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local",
  "num_samples": 64,
  "num_spatial_tokens_per_bin": 196,
  "num_temporal_bins": 8,
  "raw_token_shapes": {
    "context_example": [
      1,
      1568,
      768
    ],
    "current_example": [
      1,
      1568,
      768
    ],
    "future_example": [
      1,
      1568,
      768
    ]
  },
  "reason": null,
  "safe_stop": false,
  "safety_gate_pass": true,
  "save_raw_images": false,
  "save_video": false,
  "stage": "bridgedata_v2_tfds_true_temporal_step33a",
  "summary_shapes": {
    "context_summary": [
      768
    ],
    "current_summary": [
      768
    ],
    "future_summary": [
      768
    ]
  },
  "temporal_bins_available": true,
  "token_extraction_performed": true,
  "token_manifest_jsonl": "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_true_temporal_step33a_v1/true_temporal_token_manifest.jsonl",
  "tokenization_mode": "true_temporal_clip",
  "training_performed": false,
  "true_temporal_token_dir": "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_true_temporal_step33a_v1/true_temporal_token_smoke",
  "videomae_training_performed": false
}
```
