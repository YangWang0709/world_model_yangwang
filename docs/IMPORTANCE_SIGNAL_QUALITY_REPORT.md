# Importance Signal Quality Report

Command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_importance_signal_quality.py`

- structured token shard path: `/home/ubuntu22/tgpawb_world_model/data/token_shards/structured_toy`
- structured teacher checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/teacher_structured_toy_tiny_v1/checkpoints/teacher_world_model_step_000100.pt`
- structured importance output path: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/structured_toy_teacher`
- generated importance shard count: `4`
- generated importance shard files: `['importance_shard_000000.pt', 'importance_shard_000001.pt', 'importance_shard_000002.pt', 'importance_shard_000003.pt']`

## Training Summary

```json
{
  "num_steps": 100,
  "initial_loss": 0.3507133722305298,
  "final_loss": 0.003032910404726863,
  "best_loss": 0.00254505081102252,
  "loss_decreased": true,
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_structured_toy_tiny_v1/checkpoints/teacher_world_model_step_000100.pt",
  "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_structured_toy_tiny_v1/metrics.jsonl",
  "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_structured_toy_tiny_v1/summary.json",
  "device": "cuda",
  "dataset_size": 32,
  "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/teacher_structured_toy_tiny_v1"
}
```

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
  "importance_mean": 0.0005309324478730559,
  "importance_std": 0.00401536887511611,
  "importance_min": -2.428889274597168e-06,
  "importance_max": 0.0474613681435585,
  "importance_norm_mean": 0.013094807043671608,
  "importance_norm_std": 0.09739396721124649,
  "importance_norm_min": 0.0,
  "importance_norm_max": 1.0,
  "base_loss_mean": 0.002614582423120737,
  "masked_loss_mean": 0.003145514987409115,
  "split": "structured_toy",
  "source_token_shard": "/home/ubuntu22/tgpawb_world_model/data/token_shards/structured_toy/tokens_shard_000000.pt"
}
```

## Quality Metrics

```json
{
  "num_samples": 32,
  "num_tokens": 196,
  "num_key_tokens": 4,
  "importance_mean": 0.000511441205162555,
  "importance_std": 0.003880133619531989,
  "importance_min": -3.086170181632042e-06,
  "importance_max": 0.04816172271966934,
  "positive_importance_ratio": 0.6173469424247742,
  "key_token_importance_mean": 0.025051988661289215,
  "non_key_token_importance_mean": 1.7984150701977342e-07,
  "key_vs_non_key_gap": 0.025051808819782195,
  "top1_hit_rate": 1.0,
  "topk_hit_rate": 1.0,
  "auc_like_rank_score": 1.0,
  "score_field": "importance_scores",
  "normalized_importance_mean": 0.012825509533286095,
  "normalized_importance_std": 0.09571697562932968,
  "normalized_importance_min": 0.0,
  "normalized_importance_max": 1.0,
  "normalized_topk_hit_rate": 1.0,
  "token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/structured_toy",
  "importance_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/structured_toy_teacher",
  "importance_shard_files": [
    "importance_shard_000000.pt",
    "importance_shard_000001.pt",
    "importance_shard_000002.pt",
    "importance_shard_000003.pt"
  ],
  "quality_gates": {
    "positive_importance_ratio": true,
    "importance_std": true,
    "normalized_importance_std": true,
    "key_gt_non_key": true,
    "topk_hit_rate": true
  },
  "pass": true
}
```

IMPORTANCE_SIGNAL_QUALITY_PASS = true
