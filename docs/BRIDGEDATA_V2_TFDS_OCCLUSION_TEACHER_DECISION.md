# BridgeData V2 TFDS Occlusion Teacher Decision

Recommended Step31:
diagnose teacher architecture / temporal tokens / horizon before selector training

Do not claim final context utility.
Do not train current importance yet.
Do not train selector yet.

- condition: `teacher_topK does not beat proxy_topK`
- scope: `no selector/current-importance training yet`
- context_utility_claim_allowed: `false`
- selector_training_allowed: `false`
- current_importance_training_allowed: `false`
