# BridgeData V2 TFDS Train-Val Context Utility Report

- pass: `true`
- safe_stop: `false`
- num_selected_windows: `32`
- num_train_windows: `24`
- num_val_windows: `8`
- train_val_trajectory_disjoint: `true`
- limited_token_extraction_performed: `true`
- token_shapes: `{'context': [16, 392, 768], 'current': [4, 392, 768], 'future': [4, 392, 768]}`
- limited_proxy_importance_generation_performed: `true`
- importance_shapes: `{'context': [16, 392], 'temporal': [16], 'spatial': [392]}`
- tiny_trainval_training_performed: `true`
- optimizer_step_performed: `true`
- optimizer_step_scope: `tiny_world_model_predictor_only`
- context_utility_sanity_signal: `inconclusive`
- context_utility_claim_allowed: `false`
- current_tokens_kept_full: `true`
- train_current_importance: `false`
- new_tfds_shard_downloaded: `false`
- model_download_performed: `false`
- data_token_shards_written: `false`
- data_importance_shards_written: `false`
- safety_gate_pass: `true`

## Policy Train/Val Table

```json
[
  {
    "current_tokens_kept_full": true,
    "mean_selected_importance_mass_train": 0.0,
    "mean_selected_importance_mass_val": 0.0,
    "num_curve_points": 22,
    "num_train_windows": 24,
    "num_val_windows": 8,
    "optimizer_step_performed": true,
    "optimizer_step_scope": "tiny_world_model_predictor_only",
    "policy": "current_only",
    "topk": 0,
    "train_best_loss": 0.0006314211641438305,
    "train_current_importance": false,
    "train_final_loss": 0.0006314211641438305,
    "train_initial_loss": 13.474696159362793,
    "train_loss_decreased": true,
    "train_relative_loss_decrease": 0.9999531402299038,
    "val_best_loss": 3.9756932258605957,
    "val_final_loss": 4.146309852600098,
    "val_initial_loss": 12.292689323425293,
    "val_loss_finite": true,
    "val_relative_loss_decrease": 0.6627011597292406
  },
  {
    "current_tokens_kept_full": true,
    "mean_selected_importance_mass_train": 0.04022961853736463,
    "mean_selected_importance_mass_val": 0.03771068050638235,
    "num_curve_points": 22,
    "num_train_windows": 24,
    "num_val_windows": 8,
    "optimizer_step_performed": true,
    "optimizer_step_scope": "tiny_world_model_predictor_only",
    "policy": "random_context_topk",
    "topk": 256,
    "train_best_loss": 0.0017314744181931019,
    "train_current_importance": false,
    "train_final_loss": 0.003286305582150817,
    "train_initial_loss": 13.921120643615723,
    "train_loss_decreased": true,
    "train_relative_loss_decrease": 0.9997639338336128,
    "val_best_loss": 4.1545281410217285,
    "val_final_loss": 4.206374168395996,
    "val_initial_loss": 12.865137100219727,
    "val_loss_finite": true,
    "val_relative_loss_decrease": 0.6730408595238248
  },
  {
    "current_tokens_kept_full": true,
    "mean_selected_importance_mass_train": 0.22135961187263842,
    "mean_selected_importance_mass_val": 0.2331920640632571,
    "num_curve_points": 22,
    "num_train_windows": 24,
    "num_val_windows": 8,
    "optimizer_step_performed": true,
    "optimizer_step_scope": "tiny_world_model_predictor_only",
    "policy": "proxy_importance_topk",
    "topk": 256,
    "train_best_loss": 0.0015795632498338819,
    "train_current_importance": false,
    "train_final_loss": 0.004441520199179649,
    "train_initial_loss": 13.896537780761719,
    "train_loss_decreased": true,
    "train_relative_loss_decrease": 0.9996803865632397,
    "val_best_loss": 4.098171234130859,
    "val_final_loss": 4.098171234130859,
    "val_initial_loss": 12.576406478881836,
    "val_loss_finite": true,
    "val_relative_loss_decrease": 0.6741381378685188
  },
  {
    "current_tokens_kept_full": true,
    "mean_selected_importance_mass_train": 1.0,
    "mean_selected_importance_mass_val": 1.0,
    "num_curve_points": 22,
    "num_train_windows": 24,
    "num_val_windows": 8,
    "optimizer_step_performed": true,
    "optimizer_step_scope": "tiny_world_model_predictor_only",
    "policy": "full_context_reference",
    "topk": null,
    "train_best_loss": 0.0009924389887601137,
    "train_current_importance": false,
    "train_final_loss": 0.0009924389887601137,
    "train_initial_loss": 13.790861129760742,
    "train_loss_decreased": true,
    "train_relative_loss_decrease": 0.9999280364743418,
    "val_best_loss": 4.118686199188232,
    "val_final_loss": 4.365049362182617,
    "val_initial_loss": 12.358785629272461,
    "val_loss_finite": true,
    "val_relative_loss_decrease": 0.6468059651553661
  }
]
```

## Context Utility Decision

```json
{
  "all_val_losses_finite": true,
  "allowed_claim": "train-val sanity only; not final paper result",
  "context_utility_claim_allowed": false,
  "context_utility_sanity_signal": "inconclusive",
  "context_utility_signal": "inconclusive",
  "current_only_val_final_loss": 4.146309852600098,
  "do_not_train_current_importance_yet": true,
  "do_not_train_selector_yet": true,
  "full_better_than_current_only": false,
  "full_context_reference_val_final_loss": 4.365049362182617,
  "full_val_relative_improvement_over_current_only": -0.05275522509379052,
  "never_claim_final_utility": true,
  "proxy_better_than_current_only": true,
  "proxy_better_than_random": true,
  "proxy_importance_topk_val_final_loss": 4.098171234130859,
  "proxy_val_relative_improvement_over_current_only": 0.011609990613473127,
  "proxy_val_relative_improvement_over_random": 0.025723563794705744,
  "random_context_topk_val_final_loss": 4.206374168395996,
  "reason": "held-out policy ordering is mixed and does not justify a context utility claim.",
  "recommended_step30": {
    "condition": "Step29 context utility sanity is inconclusive",
    "name": "improve teacher label or increase window diversity before selector training",
    "scope": "no selector training yet; no current importance training yet"
  },
  "stage": "bridgedata_v2_tfds_trainval_context_utility_step29",
  "train_current_importance": false
}
```
