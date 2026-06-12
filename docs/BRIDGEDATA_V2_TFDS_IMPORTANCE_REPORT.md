# BridgeData V2 TFDS Importance Report

- pass: `true`
- safe_stop: `false`
- importance_generation_performed: `true`
- method: `proxy_token_mse_dryrun`
- num_samples: `4`
- context_importance_shape_example: `[16, 392]`
- temporal_importance_shape_example: `[16]`
- spatial_importance_shape_example: `[392]`
- importance_norm_min: `0.0`
- importance_norm_max: `1.0`
- current_tokens_kept_full: `true`
- train_current_importance: `false`
- download_performed: `false`
- training_performed: `false`
- token_extraction_performed: `false`
- large_importance_shards_generated: `false`
- data_importance_shards_written: `false`
- safety_gate_pass: `true`
- label_quality_note: `proxy dry-run only; not final teacher label`
- recommended_step26: `BridgeData V2 TFDS mini-shard context bottleneck world-model smoke`

## Summary JSON

```json
{
  "action_used_as_input": false,
  "context_importance_shape_example": [
    16,
    392
  ],
  "current_importance_generated": false,
  "current_tokens_kept_full": true,
  "data_importance_shards_written": false,
  "data_token_shards_written": false,
  "download_performed": false,
  "env_isaaclab_guard": {
    "tensorflow": false,
    "tensorflow_datasets": false
  },
  "goal_used_as_input": false,
  "importance_dir": "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_importance_step25_v1/importance_smoke",
  "importance_generation_performed": true,
  "importance_manifest_jsonl": "/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_importance_step25_v1/tfds_importance_smoke_manifest.jsonl",
  "importance_norm_max": 1.0,
  "importance_norm_min": 0.0,
  "importance_raw_mean": 5.381548930927238e-06,
  "importance_raw_std": 8.345384685526369e-06,
  "label_quality_note": "proxy dry-run only; not final teacher label",
  "language_used_as_input": false,
  "large_importance_shards_generated": false,
  "method": "proxy_token_mse_dryrun",
  "model_download_performed": false,
  "num_importance_artifacts": 4,
  "num_samples": 4,
  "reason": null,
  "safe_stop": false,
  "safety_gate_pass": true,
  "selector_training_performed": false,
  "spatial_importance_shape_example": [
    392
  ],
  "stage": "bridgedata_v2_tfds_importance_step25",
  "teacher_training_performed": false,
  "temporal_importance_shape_example": [
    16
  ],
  "token_extraction_performed": false,
  "train_current_importance": false,
  "training_performed": false,
  "world_model_training_performed": false
}
```
