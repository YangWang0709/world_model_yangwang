# BridgeData V2 TFDS Window Diversity Report

- pass: `true`
- safe_stop: `false`
- num_selected_windows: `64`
- num_selected_trajectories: `10`
- split_seeds: `[42, 123, 999]`
- limited_token_extraction_performed: `true`
- token_shapes: `{'context': [16, 392, 768], 'current': [4, 392, 768], 'future': [4, 392, 768]}`
- limited_proxy_importance_generation_performed: `true`
- importance_shapes: `{'context': [16, 392], 'temporal': [16], 'spatial': [392]}`
- tiny_trainval_training_performed: `true`
- optimizer_step_scope: `tiny_world_model_predictor_only`
- proxy_beats_current_fraction: `1.0`
- proxy_beats_random_fraction: `1.0`
- full_beats_current_fraction: `0.0`
- context_signal_stability: `weak_proxy_positive_full_context_noisy`
- context_utility_claim_allowed: `false`
- selector_training_allowed: `false`
- current_importance_training_allowed: `false`
- new_tfds_shard_downloaded: `false`
- model_download_performed: `false`
- data_token_shards_written: `false`
- data_importance_shards_written: `false`
- safety_gate_pass: `true`

## Split Summary

```json
[
  {
    "fallback_reason": null,
    "fallback_used": false,
    "num_train_windows": 48,
    "num_val_windows": 16,
    "split_seed": 42,
    "trajectory_disjoint": true
  },
  {
    "fallback_reason": null,
    "fallback_used": false,
    "num_train_windows": 48,
    "num_val_windows": 16,
    "split_seed": 123,
    "trajectory_disjoint": true
  },
  {
    "fallback_reason": null,
    "fallback_used": false,
    "num_train_windows": 48,
    "num_val_windows": 16,
    "split_seed": 999,
    "trajectory_disjoint": true
  }
]
```

## Per-Seed Validation Table

```json
[
  {
    "current_only": 3.833101749420166,
    "full_context_reference": 4.072451591491699,
    "num_train_windows": 48,
    "num_val_windows": 16,
    "proxy_importance_topk": 3.4459614753723145,
    "random_context_topk": 3.696502685546875,
    "split_seed": 42,
    "trajectory_disjoint": true
  },
  {
    "current_only": 3.828800916671753,
    "full_context_reference": 3.8221139907836914,
    "num_train_windows": 48,
    "num_val_windows": 16,
    "proxy_importance_topk": 3.5001325607299805,
    "random_context_topk": 3.8678016662597656,
    "split_seed": 123,
    "trajectory_disjoint": true
  },
  {
    "current_only": 3.1345431804656982,
    "full_context_reference": 3.2998008728027344,
    "num_train_windows": 48,
    "num_val_windows": 16,
    "proxy_importance_topk": 2.774118185043335,
    "random_context_topk": 3.325176954269409,
    "split_seed": 999,
    "trajectory_disjoint": true
  }
]
```

## Stability Summary

```json
{
  "aggregate_metrics": {
    "all_val_losses_finite": true,
    "full_beats_current_count": 0,
    "num_runs": 3,
    "policy_val_best_mean": {
      "current_only": 3.244630972544352,
      "full_context_reference": 3.3072267373402915,
      "proxy_importance_topk": 3.18841822942098,
      "random_context_topk": 3.276494820912679
    },
    "policy_val_final_mean": {
      "current_only": 3.5988152821858725,
      "full_context_reference": 3.7314554850260415,
      "proxy_importance_topk": 3.2400707403818765,
      "random_context_topk": 3.62982710202535
    },
    "policy_val_final_values": {
      "current_only": [
        3.833101749420166,
        3.828800916671753,
        3.1345431804656982
      ],
      "full_context_reference": [
        4.072451591491699,
        3.8221139907836914,
        3.2998008728027344
      ],
      "proxy_importance_topk": [
        3.4459614753723145,
        3.5001325607299805,
        2.774118185043335
      ],
      "random_context_topk": [
        3.696502685546875,
        3.8678016662597656,
        3.325176954269409
      ]
    },
    "proxy_beats_current_count": 3,
    "proxy_beats_random_count": 3
  },
  "context_signal_stability": "weak_proxy_positive_full_context_noisy",
  "context_utility_claim_allowed": false,
  "current_importance_training_allowed": false,
  "full_beats_current_fraction": 0.0,
  "mean_full_improvement_over_current": -0.03780594699751538,
  "mean_proxy_improvement_over_current": 0.10060837979211541,
  "mean_proxy_improvement_over_random": 0.10951999373163708,
  "num_seeds": 3,
  "proxy_beats_current_and_random_fraction": 1.0,
  "proxy_beats_current_fraction": 1.0,
  "proxy_beats_random_fraction": 1.0,
  "recommended_step30b": {
    "condition": "proxy top-k is stable but full context is noisy",
    "name": "trained-predictor occlusion teacher or add more data diversity",
    "scope": "no selector/current-importance training yet"
  },
  "safety_gate_pass": true,
  "seed_summaries": [
    {
      "full_better_than_current_only": false,
      "full_val_relative_improvement_over_current_only": -0.062442861608810594,
      "proxy_better_than_current_only": true,
      "proxy_better_than_random": true,
      "proxy_val_relative_improvement_over_current_only": 0.10099921665434901,
      "proxy_val_relative_improvement_over_random": 0.06777790562797725,
      "split_seed": 42
    },
    {
      "full_better_than_current_only": false,
      "full_val_relative_improvement_over_current_only": 0.0017464804343690563,
      "proxy_better_than_current_only": true,
      "proxy_better_than_random": true,
      "proxy_val_relative_improvement_over_current_only": 0.08584106698017424,
      "proxy_val_relative_improvement_over_random": 0.09505893457182561,
      "split_seed": 123
    },
    {
      "full_better_than_current_only": false,
      "full_val_relative_improvement_over_current_only": -0.05272145981810461,
      "proxy_better_than_current_only": true,
      "proxy_better_than_random": true,
      "proxy_val_relative_improvement_over_current_only": 0.11498485574182297,
      "proxy_val_relative_improvement_over_random": 0.16572314099510835,
      "split_seed": 999
    }
  ],
  "selector_training_allowed": false,
  "stage": "bridgedata_v2_tfds_window_diversity_step30a"
}
```

## Context Signal Decision

```json
{
  "context_signal_stability": "weak_proxy_positive_full_context_noisy",
  "context_utility_claim_allowed": false,
  "current_importance_training_allowed": false,
  "do_not_train_current_importance_yet": true,
  "do_not_train_selector_yet": true,
  "never_claim_final_utility": true,
  "reason": "proxy top-k is stable across seeds while full context remains noisy.",
  "recommended_step30b": {
    "condition": "proxy top-k is stable but full context is noisy",
    "name": "trained-predictor occlusion teacher or add more data diversity",
    "scope": "no selector/current-importance training yet"
  },
  "safety_gate_pass": true,
  "selector_training_allowed": false,
  "stage": "bridgedata_v2_tfds_window_diversity_step30a"
}
```
