# BridgeData V2 Longer-Horizon Window Report

- longer_horizon_window_builder_performed: `true`
- horizons_completed: `[0, 4, 8, 12]`
- selected_windows_by_horizon: `{'gap0': 64, 'gap12': 63, 'gap4': 64, 'gap8': 64}`
- trajectory_count_by_horizon: `{'gap0': 10, 'gap12': 5, 'gap4': 8, 'gap8': 7}`
- split_seeds: `[42, 123, 999]`
- limited_clip_export_performed: `true`
- limited_token_extraction_performed: `true`
- token_shapes: `{'context': [16, 392, 768], 'current': [4, 392, 768], 'future': [4, 392, 768]}`
- limited_proxy_importance_generation_performed: `true`
- importance_shapes: `{'context': [16, 392], 'temporal': [16], 'spatial': [392]}`
- data_token_shards_written: `false`
- data_importance_shards_written: `false`

## Split Summary

```json
[
  {
    "fallback_reason": null,
    "fallback_used": false,
    "horizon_gap": 0,
    "num_train_windows": 41,
    "num_val_windows": 23,
    "split_seed": 42,
    "trajectory_disjoint": true
  },
  {
    "fallback_reason": null,
    "fallback_used": false,
    "horizon_gap": 0,
    "num_train_windows": 46,
    "num_val_windows": 18,
    "split_seed": 123,
    "trajectory_disjoint": true
  },
  {
    "fallback_reason": null,
    "fallback_used": false,
    "horizon_gap": 0,
    "num_train_windows": 45,
    "num_val_windows": 19,
    "split_seed": 999,
    "trajectory_disjoint": true
  },
  {
    "fallback_reason": null,
    "fallback_used": false,
    "horizon_gap": 4,
    "num_train_windows": 44,
    "num_val_windows": 20,
    "split_seed": 42,
    "trajectory_disjoint": true
  },
  {
    "fallback_reason": null,
    "fallback_used": false,
    "horizon_gap": 4,
    "num_train_windows": 43,
    "num_val_windows": 21,
    "split_seed": 123,
    "trajectory_disjoint": true
  },
  {
    "fallback_reason": null,
    "fallback_used": false,
    "horizon_gap": 4,
    "num_train_windows": 48,
    "num_val_windows": 16,
    "split_seed": 999,
    "trajectory_disjoint": true
  },
  {
    "fallback_reason": null,
    "fallback_used": false,
    "horizon_gap": 8,
    "num_train_windows": 48,
    "num_val_windows": 16,
    "split_seed": 42,
    "trajectory_disjoint": true
  },
  {
    "fallback_reason": null,
    "fallback_used": false,
    "horizon_gap": 8,
    "num_train_windows": 38,
    "num_val_windows": 26,
    "split_seed": 123,
    "trajectory_disjoint": true
  },
  {
    "fallback_reason": null,
    "fallback_used": false,
    "horizon_gap": 8,
    "num_train_windows": 41,
    "num_val_windows": 23,
    "split_seed": 999,
    "trajectory_disjoint": true
  },
  {
    "fallback_reason": null,
    "fallback_used": false,
    "horizon_gap": 12,
    "num_train_windows": 39,
    "num_val_windows": 24,
    "split_seed": 42,
    "trajectory_disjoint": true
  },
  {
    "fallback_reason": null,
    "fallback_used": false,
    "horizon_gap": 12,
    "num_train_windows": 37,
    "num_val_windows": 26,
    "split_seed": 123,
    "trajectory_disjoint": true
  },
  {
    "fallback_reason": null,
    "fallback_used": false,
    "horizon_gap": 12,
    "num_train_windows": 38,
    "num_val_windows": 25,
    "split_seed": 999,
    "trajectory_disjoint": true
  }
]
```
