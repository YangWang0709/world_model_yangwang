# BridgeData V2 TFDS Train-Val Context Utility Decision

Recommended Step30:
improve teacher label or increase window diversity before selector training.

Do not claim final context utility.
Do not train current importance yet.
Do not train selector yet.

- context_utility_sanity_signal: `inconclusive`
- context_utility_claim_allowed: `false`
- recommended_step30_name: `improve teacher label or increase window diversity before selector training`
- recommended_step30_condition: `Step29 context utility sanity is inconclusive`
- recommended_step30_scope: `no selector training yet; no current importance training yet`
