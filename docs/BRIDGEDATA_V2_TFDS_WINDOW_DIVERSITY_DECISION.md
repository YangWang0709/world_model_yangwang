# BridgeData V2 TFDS Window Diversity Decision

Recommended Step30B:
trained-predictor occlusion teacher on expanded windows.

Do not claim final context utility.
Do not train current importance yet.
Do not train selector yet.

- context_signal_stability: `weak_proxy_positive_full_context_noisy`
- context_utility_claim_allowed: `false`
- selector_training_allowed: `false`
- current_importance_training_allowed: `false`
- recommended_step30b_name: `trained-predictor occlusion teacher or add more data diversity`
- recommended_step30b_condition: `proxy top-k is stable but full context is noisy`
- recommended_step30b_scope: `no selector/current-importance training yet`
