# BridgeData V2 True-Temporal Target Diagnosis Report

- target_variant: `future_delta_last_minus_current`
- frame_repeat_proxy_gain_mean: `0.49523736761231846`
- true_temporal_proxy_gain_mean: `0.1919530656957297`
- true_temporal_gain_over_frame_repeat: `-0.3032843019165887`
- true_temporal_proxy_beats_current_fraction: `1.0`
- true_temporal_proxy_beats_random_fraction: `1.0`
- full_context_noise_confirmed: `true`
- true_temporal_representation_helped: `false`
- context_utility_claim_allowed: `false`
- selector_training_allowed: `false`
- current_importance_training_allowed: `false`

## Target Rows

```json
[
  {
    "full_beats_current_fraction": 1.0,
    "full_context_noise_penalty_present": true,
    "horizon_gap": 0,
    "mean_policy_val": {
      "current_only": 1.3410722017288208,
      "full_context_reference": 1.1312851905822754,
      "proxy_importance_topk": 1.0800968607266743,
      "random_context_topk": 1.1806879838307698
    },
    "mean_proxy_gain_over_current": 0.1919530656957297,
    "num_rows": 3,
    "proxy_beats_current_fraction": 1.0,
    "proxy_beats_random_fraction": 1.0,
    "representation_mode": "true_temporal_clip_step33a",
    "target_variant": "future_delta_last_minus_current"
  },
  {
    "full_beats_current_fraction": 1.0,
    "full_context_noise_penalty_present": true,
    "horizon_gap": 0,
    "mean_policy_val": {
      "current_only": 1.3771350781122844,
      "full_context_reference": 1.1785997152328491,
      "proxy_importance_topk": 1.1157835721969604,
      "random_context_topk": 1.129799485206604
    },
    "mean_proxy_gain_over_current": 0.18669520104161383,
    "num_rows": 3,
    "proxy_beats_current_fraction": 1.0,
    "proxy_beats_random_fraction": 0.6666666666666666,
    "representation_mode": "true_temporal_clip_step33a",
    "target_variant": "future_delta_mean_minus_current"
  },
  {
    "full_beats_current_fraction": 0.6666666666666666,
    "full_context_noise_penalty_present": true,
    "horizon_gap": 0,
    "mean_policy_val": {
      "current_only": 2.645601749420166,
      "full_context_reference": 2.52852725982666,
      "proxy_importance_topk": 2.3257644176483154,
      "random_context_topk": 2.4943849245707193
    },
    "mean_proxy_gain_over_current": 0.11955502973636807,
    "num_rows": 3,
    "proxy_beats_current_fraction": 1.0,
    "proxy_beats_random_fraction": 1.0,
    "representation_mode": "true_temporal_clip_step33a",
    "target_variant": "future_mean_all4"
  }
]
```

## Per-Seed Validation Table

```json
[
  {
    "current_only": 1.3815081119537354,
    "full_beats_current": true,
    "full_context_reference": 1.1121089458465576,
    "horizon_gap": 0,
    "num_train_windows": 41,
    "num_val_windows": 23,
    "proxy_beats_current": true,
    "proxy_beats_random": true,
    "proxy_gain_over_current": 0.22371193889340948,
    "proxy_importance_topk": 1.0724482536315918,
    "random_context_topk": 1.089299201965332,
    "representation_mode": "true_temporal_clip_step33a",
    "split_seed": 42,
    "target_variant": "future_delta_last_minus_current",
    "trajectory_disjoint": true
  },
  {
    "current_only": 1.418036937713623,
    "full_beats_current": true,
    "full_context_reference": 1.162097454071045,
    "horizon_gap": 0,
    "num_train_windows": 41,
    "num_val_windows": 23,
    "proxy_beats_current": true,
    "proxy_beats_random": true,
    "proxy_gain_over_current": 0.25751192314042554,
    "proxy_importance_topk": 1.0528755187988281,
    "random_context_topk": 1.0975974798202515,
    "representation_mode": "true_temporal_clip_step33a",
    "split_seed": 42,
    "target_variant": "future_delta_mean_minus_current",
    "trajectory_disjoint": true
  },
  {
    "current_only": 2.6076736450195312,
    "full_beats_current": true,
    "full_context_reference": 2.4230599403381348,
    "horizon_gap": 0,
    "num_train_windows": 41,
    "num_val_windows": 23,
    "proxy_beats_current": true,
    "proxy_beats_random": true,
    "proxy_gain_over_current": 0.1200111434406205,
    "proxy_importance_topk": 2.2947237491607666,
    "random_context_topk": 2.35961651802063,
    "representation_mode": "true_temporal_clip_step33a",
    "split_seed": 42,
    "target_variant": "future_mean_all4",
    "trajectory_disjoint": true
  },
  {
    "current_only": 1.362184762954712,
    "full_beats_current": true,
    "full_context_reference": 1.031069040298462,
    "horizon_gap": 0,
    "num_train_windows": 46,
    "num_val_windows": 18,
    "proxy_beats_current": true,
    "proxy_beats_random": true,
    "proxy_gain_over_current": 0.2816973187848883,
    "proxy_importance_topk": 0.978460967540741,
    "random_context_topk": 1.0700067281723022,
    "representation_mode": "true_temporal_clip_step33a",
    "split_seed": 123,
    "target_variant": "future_delta_last_minus_current",
    "trajectory_disjoint": true
  },
  {
    "current_only": 1.4104377031326294,
    "full_beats_current": true,
    "full_context_reference": 1.098086953163147,
    "horizon_gap": 0,
    "num_train_windows": 46,
    "num_val_windows": 18,
    "proxy_beats_current": true,
    "proxy_beats_random": false,
    "proxy_gain_over_current": 0.22938570738772723,
    "proxy_importance_topk": 1.08690345287323,
    "random_context_topk": 1.0544496774673462,
    "representation_mode": "true_temporal_clip_step33a",
    "split_seed": 123,
    "target_variant": "future_delta_mean_minus_current",
    "trajectory_disjoint": true
  },
  {
    "current_only": 2.9532296657562256,
    "full_beats_current": true,
    "full_context_reference": 2.7527034282684326,
    "horizon_gap": 0,
    "num_train_windows": 46,
    "num_val_windows": 18,
    "proxy_beats_current": true,
    "proxy_beats_random": true,
    "proxy_gain_over_current": 0.13777915631251897,
    "proxy_importance_topk": 2.5463361740112305,
    "random_context_topk": 2.771811008453369,
    "representation_mode": "true_temporal_clip_step33a",
    "split_seed": 123,
    "target_variant": "future_mean_all4",
    "trajectory_disjoint": true
  },
  {
    "current_only": 1.2795237302780151,
    "full_beats_current": true,
    "full_context_reference": 1.2506775856018066,
    "horizon_gap": 0,
    "num_train_windows": 45,
    "num_val_windows": 19,
    "proxy_beats_current": true,
    "proxy_beats_random": true,
    "proxy_gain_over_current": 0.07044993940889127,
    "proxy_importance_topk": 1.1893813610076904,
    "random_context_topk": 1.3827580213546753,
    "representation_mode": "true_temporal_clip_step33a",
    "split_seed": 999,
    "target_variant": "future_delta_last_minus_current",
    "trajectory_disjoint": true
  },
  {
    "current_only": 1.3029305934906006,
    "full_beats_current": true,
    "full_context_reference": 1.2756147384643555,
    "horizon_gap": 0,
    "num_train_windows": 45,
    "num_val_windows": 19,
    "proxy_beats_current": true,
    "proxy_beats_random": true,
    "proxy_gain_over_current": 0.07318797259668865,
    "proxy_importance_topk": 1.2075717449188232,
    "random_context_topk": 1.2373512983322144,
    "representation_mode": "true_temporal_clip_step33a",
    "split_seed": 999,
    "target_variant": "future_delta_mean_minus_current",
    "trajectory_disjoint": true
  },
  {
    "current_only": 2.375901937484741,
    "full_beats_current": false,
    "full_context_reference": 2.409818410873413,
    "horizon_gap": 0,
    "num_train_windows": 45,
    "num_val_windows": 19,
    "proxy_beats_current": true,
    "proxy_beats_random": true,
    "proxy_gain_over_current": 0.10087478945596474,
    "proxy_importance_topk": 2.136233329772949,
    "random_context_topk": 2.351727247238159,
    "representation_mode": "true_temporal_clip_step33a",
    "split_seed": 999,
    "target_variant": "future_mean_all4",
    "trajectory_disjoint": true
  }
]
```
