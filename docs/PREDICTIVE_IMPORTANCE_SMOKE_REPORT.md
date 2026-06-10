# Predictive Importance Smoke Report

Command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_predictive_importance.py`

- teacher checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/teacher_dummy_tiny_v1/checkpoints/teacher_world_model_step_000030.pt`
- token shard input dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/dummy_toy`
- importance output dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/dummy_toy_teacher`
- generated importance shard count: `2`
- generated importance shard files: `['importance_shard_000000.pt', 'importance_shard_000001.pt']`
- validation result: `True`

## First Importance Shard Summary

```json
{
  "schema_version": "0.1.0",
  "importance_method": "teacher_token_occlusion",
  "num_samples": 8,
  "num_tokens": 196,
  "importance_scores_shape": [
    8,
    196
  ],
  "importance_scores_norm_shape": [
    8,
    196
  ],
  "base_losses_shape": [
    8
  ],
  "masked_losses_shape": [
    8,
    196
  ],
  "importance_mean": -4.0673327816875826e-07,
  "importance_std": 9.629547292888674e-08,
  "importance_min": -5.339461495168507e-07,
  "importance_max": -2.8860813472419977e-07,
  "importance_norm_mean": 0.0,
  "importance_norm_std": 0.0,
  "importance_norm_min": 0.0,
  "importance_norm_max": 0.0,
  "base_loss_mean": 8.536633686162531e-05,
  "masked_loss_mean": 8.495961810695007e-05,
  "split": "toy",
  "source_token_shard": "/home/ubuntu22/tgpawb_world_model/data/token_shards/dummy_toy/tokens_shard_000000.pt"
}
```

## Eval Importance Summary

```json
{
  "importance_dir": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/dummy_toy_teacher",
  "num_shards": 2,
  "num_samples": 16,
  "num_tokens": 196,
  "shard_files": [
    "importance_shard_000000.pt",
    "importance_shard_000001.pt"
  ],
  "importance_mean": -4.121299923554034e-07,
  "importance_std": 9.619645169323121e-08,
  "importance_min": -5.339461495168507e-07,
  "importance_max": -2.8860813472419977e-07,
  "normalized_importance_mean": 0.0,
  "normalized_importance_std": 0.0,
  "normalized_importance_min": 0.0,
  "normalized_importance_max": 0.0,
  "base_loss_mean": 8.54049576446414e-05,
  "masked_loss_mean": 8.49928183015436e-05,
  "top1_importance_mean": -4.118696779187303e-07,
  "top5_importance_mean": -4.1191307786903053e-07
}
```

## Generation Summary

```json
{
  "num_source_shards": 2,
  "num_importance_shards": 2,
  "device": "cuda",
  "importance_shard_files": [
    "importance_shard_000000.pt",
    "importance_shard_000001.pt"
  ]
}
```

PREDICTIVE_IMPORTANCE_SMOKE_PASS = true
