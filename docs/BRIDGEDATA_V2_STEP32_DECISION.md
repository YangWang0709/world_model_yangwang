# BridgeData V2 Step32 Decision

Recommended Step33:
diagnose data diversity or representation before selector training

Do not claim final context utility.
Do not train selector yet.
Do not train current importance yet.

- condition: `longer horizon does not increase proxy context gain`
- scope: `no selector/current-importance training yet`
- best_horizon_gap: `0`
- best_target_variant: `future_delta_last_minus_current`
- delta_target_amplifies_context_gain: `false`
- context_utility_claim_allowed: `false`
- selector_training_allowed: `false`
- current_importance_training_allowed: `false`
