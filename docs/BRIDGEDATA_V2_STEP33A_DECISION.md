# BridgeData V2 Step33A Decision

Recommended Step33B/Step34:
add data diversity with another TFDS shard or diagnose dataset bias

Do not claim final context utility.
Do not train selector yet.
Do not train current importance yet.

- condition: `true temporal representation underperforms frame-repeat baseline`
- scope: `no selector/current-importance training yet`
- true_temporal_representation_helped: `false`
- frame_repeat_proxy_gain_mean: `0.49523736761231846`
- true_temporal_proxy_gain_mean: `0.1919530656957297`
- true_temporal_gain_over_frame_repeat: `-0.3032843019165887`
- context_utility_claim_allowed: `false`
- selector_training_allowed: `false`
- current_importance_training_allowed: `false`
