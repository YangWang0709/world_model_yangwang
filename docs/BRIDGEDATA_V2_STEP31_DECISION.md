# BridgeData V2 Step31 Decision

Recommended Step32:
longer-horizon BridgeData window builder and target diagnosis

Do not claim final context utility.
Do not train current importance yet.
Do not train selector yet.

- condition: `delta target exposes much stronger context gain under existing tokens`
- scope: `no selector/current-importance training yet`
- best_architecture_variant: `current_plus_proxy_topk_context_predictor`
- best_diagnostic_score: `proxy_importance`
- current_dominance_level: `low`
- context_utility_claim_allowed: `false`
- selector_training_allowed: `false`
- current_importance_training_allowed: `false`
