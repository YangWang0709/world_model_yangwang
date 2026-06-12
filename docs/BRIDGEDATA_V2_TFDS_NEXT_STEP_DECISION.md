# BridgeData V2 TFDS Next Step Decision

Recommended Step29:
BridgeData V2 TFDS 32/64-window train-val context utility sanity using existing shard first.

Do not train current importance yet.
Do not train selector yet.
Do not claim context utility from 4-sample overfit.

- recommended_step29_name: `BridgeData V2 TFDS 32/64-window train-val context utility sanity`
- recommended_step29_condition: `Step28 shows 4-sample overfit is memorization-prone`
- recommended_step29_scope: `use existing shard first; allow limited token extraction expansion, not full training`
- alternative_step29_name: `trained-predictor occlusion teacher dry-run`
- alternative_step29_condition: `if user chooses teacher-first path`
- alternative_step29_scope: `no selector training yet`
- context_utility_claim_allowed: `false`
- train_current_importance_now: `false`
