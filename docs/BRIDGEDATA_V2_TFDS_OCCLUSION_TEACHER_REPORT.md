# BridgeData V2 TFDS Occlusion Teacher Report

- pass: `true`
- safe_stop: `false`
- trained_predictor_teacher_training_performed: `true`
- teacher_training_scope: `small_predictor_teacher_only`
- split_seeds_trained: `[42, 123, 999]`
- teacher_val_loss_finite: `true`
- teacher_occlusion_importance_generated: `true`
- num_samples: `64`
- teacher_label_shape: `[16, 392]`
- fallback_used_count: `9`
- teacher_label_nontrivial: `true`
- teacher_proxy_pearson_mean: `-0.006298676786173019`
- teacher_proxy_spearman_mean: `0.12641619730766251`
- teacher_beats_current_fraction: `1.0`
- teacher_beats_random_fraction: `1.0`
- teacher_beats_proxy_fraction: `0.0`
- context_utility_claim_allowed: `false`
- selector_training_allowed: `false`
- current_importance_training_allowed: `false`
- safety_gate_pass: `true`

## Teacher Train Summary

```json
[
  {
    "num_curve_points": 14,
    "num_train_windows": 48,
    "num_val_windows": 16,
    "optimizer_param_count": 6,
    "optimizer_step_scope": "small_predictor_teacher_only",
    "split_seed": 42,
    "train_best_loss": 0.04167575792719921,
    "train_final_loss": 0.05056277476251125,
    "train_initial_loss": 13.732329368591309,
    "train_relative_loss_decrease": 0.9963179753845579,
    "val_best_loss": 3.63055682182312,
    "val_final_loss": 3.8197295665740967,
    "val_initial_loss": 12.691173076629639,
    "val_loss_finite": true,
    "val_relative_loss_decrease": 0.6990247045320028
  },
  {
    "num_curve_points": 14,
    "num_train_windows": 48,
    "num_val_windows": 16,
    "optimizer_param_count": 6,
    "optimizer_step_scope": "small_predictor_teacher_only",
    "split_seed": 123,
    "train_best_loss": 0.053094894314805664,
    "train_final_loss": 0.07302874761323135,
    "train_initial_loss": 14.187211672465006,
    "train_relative_loss_decrease": 0.9948524946762466,
    "val_best_loss": 3.5775318145751953,
    "val_final_loss": 3.67402982711792,
    "val_initial_loss": 12.932707786560059,
    "val_loss_finite": true,
    "val_relative_loss_decrease": 0.7159117883312844
  },
  {
    "num_curve_points": 14,
    "num_train_windows": 48,
    "num_val_windows": 16,
    "optimizer_param_count": 6,
    "optimizer_step_scope": "small_predictor_teacher_only",
    "split_seed": 999,
    "train_best_loss": 0.05021745959917704,
    "train_final_loss": 0.08235171064734459,
    "train_initial_loss": 14.019998709360758,
    "train_relative_loss_decrease": 0.9941261256613126,
    "val_best_loss": 2.802958846092224,
    "val_final_loss": 3.17953622341156,
    "val_initial_loss": 11.861955642700195,
    "val_loss_finite": true,
    "val_relative_loss_decrease": 0.731955141362526
  }
]
```

## Teacher TopK Utility

```json
[
  {
    "current_only": 3.833101749420166,
    "full_context_reference": 3.8098902702331543,
    "proxy_importance_topk": 3.4459614753723145,
    "random_context_topk": 3.9316444396972656,
    "split_seed": 42,
    "teacher_occlusion_topk": 3.7486846446990967
  },
  {
    "current_only": 3.8279855251312256,
    "full_context_reference": 3.849149465560913,
    "proxy_importance_topk": 3.4446463584899902,
    "random_context_topk": 3.8962819576263428,
    "split_seed": 123,
    "teacher_occlusion_topk": 3.7120585441589355
  },
  {
    "current_only": 3.1035919189453125,
    "full_context_reference": 3.3621480464935303,
    "proxy_importance_topk": 2.799060821533203,
    "random_context_topk": 3.10705828666687,
    "split_seed": 999,
    "teacher_occlusion_topk": 3.037132978439331
  }
]
```
