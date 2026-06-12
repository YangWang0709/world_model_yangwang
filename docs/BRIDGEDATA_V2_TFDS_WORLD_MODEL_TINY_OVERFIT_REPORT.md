# BridgeData V2 TFDS World-Model Tiny Overfit Report

- pass: `true`
- safe_stop: `false`
- tiny_overfit_training_performed: `true`
- training_performed: `true`
- tiny_overfit_training_only: `true`
- num_samples: `4`
- policies_trained: `['current_only', 'random_context_topk', 'proxy_importance_topk', 'full_context_reference']`
- train_steps: `300`
- optimizer: `adamw`
- optimizer_step_performed: `true`
- optimizer_step_scope: `tiny_world_model_predictor_only`
- all_losses_finite: `true`
- policies_with_loss_decrease: `4`
- acceptance_pass: `true`
- current_tokens_kept_full: `true`
- train_current_importance: `false`
- current_importance_training_performed: `false`
- videomae_training_performed: `false`
- teacher_training_performed: `false`
- selector_training_performed: `false`
- world_model_large_training_performed: `false`
- token_extraction_performed: `false`
- importance_generation_performed: `false`
- download_performed: `false`
- data_token_shards_written: `false`
- data_importance_shards_written: `false`
- checkpoint_saved: `false`
- tiny_overfit_result_not_final_performance: `true`
- safety_gate_pass: `true`
- loss_quality_note: `tiny-overfit only; not final performance`
- recommended_step28: `BridgeData V2 TFDS context predictor sanity and stronger teacher planning`

## Policy Metrics

```json
[
  {
    "absolute_loss_decrease": 12.133682998130098,
    "all_losses_finite": true,
    "best_loss": 0.0011004924308508635,
    "final_loss": 0.0035450139548629522,
    "initial_loss": 12.137228012084961,
    "loss_decreased": true,
    "policy": "current_only",
    "relative_loss_decrease": 0.9997079222742349,
    "topk": 0
  },
  {
    "absolute_loss_decrease": 12.771020096726716,
    "all_losses_finite": true,
    "best_loss": 0.0021274862810969353,
    "final_loss": 0.0021274862810969353,
    "initial_loss": 12.773147583007812,
    "loss_decreased": true,
    "policy": "random_context_topk",
    "relative_loss_decrease": 0.9998334407187209,
    "topk": 256
  },
  {
    "absolute_loss_decrease": 12.600051738321781,
    "all_losses_finite": true,
    "best_loss": 0.0013253530487418175,
    "final_loss": 0.07361903041601181,
    "initial_loss": 12.673670768737793,
    "loss_decreased": true,
    "policy": "proxy_importance_topk",
    "relative_loss_decrease": 0.9941911832996634,
    "topk": 256
  },
  {
    "absolute_loss_decrease": 12.251727136783302,
    "all_losses_finite": true,
    "best_loss": 0.0011100443080067635,
    "final_loss": 0.0011100443080067635,
    "initial_loss": 12.252837181091309,
    "loss_decreased": true,
    "policy": "full_context_reference",
    "relative_loss_decrease": 0.9999094051204958,
    "topk": null
  }
]
```
