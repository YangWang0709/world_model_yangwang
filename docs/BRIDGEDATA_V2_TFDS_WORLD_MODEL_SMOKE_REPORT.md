# BridgeData V2 TFDS World-Model Smoke Report

- pass: `true`
- safe_stop: `false`
- world_model_smoke_performed: `true`
- num_samples: `4`
- policies_evaluated: `['current_only', 'random_context_topk', 'proxy_importance_topk', 'full_context_reference']`
- topk_values: `[64, 128, 256]`
- all_losses_finite: `true`
- current_tokens_kept_full: `true`
- train_current_importance: `false`
- optimizer_step_performed: `false`
- training_performed: `false`
- token_extraction_performed: `false`
- importance_generation_performed: `false`
- download_performed: `false`
- data_token_shards_written: `false`
- data_importance_shards_written: `false`
- random_init_result_not_scientific: `true`
- safety_gate_pass: `true`
- loss_quality_note: `forward-smoke only; random-init loss is not final performance`
- recommended_step27: `Step27 BridgeData V2 TFDS mini-shard tiny overfit world-model training`

## Policy Metrics

```json
[
  {
    "all_losses_finite": true,
    "mean_loss": 12.137229442596436,
    "mean_selected_importance_mass": 0.0,
    "policy": "current_only",
    "topk": 0
  },
  {
    "all_losses_finite": true,
    "mean_loss": 12.248300313949585,
    "mean_selected_importance_mass": 0.037160635033425635,
    "policy": "random_context_topk",
    "topk": 256
  },
  {
    "all_losses_finite": true,
    "mean_loss": 12.348532915115356,
    "mean_selected_importance_mass": 0.2452614637082408,
    "policy": "proxy_importance_topk",
    "topk": 256
  },
  {
    "all_losses_finite": true,
    "mean_loss": 12.242180347442627,
    "mean_selected_importance_mass": 1.0,
    "policy": "full_context_reference",
    "topk": null
  }
]
```
