# BridgeData V2 Step38B Decision

- temporal_component_dominates: `false`
- spatial_residual_learnable_evidence: `false`
- spatial_residual_cross_shard_consistent: `true`
- label_patch_detail_learnable_allowed: `false`
- selector_training_allowed: `false`
- final_selector_training_allowed: `false`
- current_importance_training_allowed: `false`
- context_utility_claim_allowed: `false`

Recommended Step39:
- name: `label redesign toward temporal-only or temporal+broadcast/coarse supervision`
- scope: `redesign proxy labels before any downstream selector use`
- reason: `spatial residual is weak, diffuse, or not cross-shard-consistent`
