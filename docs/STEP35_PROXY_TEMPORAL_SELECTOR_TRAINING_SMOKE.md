# Step35 Proxy Temporal Selector Training Smoke

Step35 is a bounded temporal selector smoke, not a final experiment.

Scope:
- Step35 trains a temporal selector only.
- Step35 does not train a patch-level selector.
- Step35 does not train the final deployable selector.
- Step35 does not train VideoMAE.
- Step35 does not train current importance.
- Step35 does not claim context utility.

Safety flags:
- optimizer_scope: `proxy_temporal_selector_head_only`
- checkpoint_saved: `false`
- selector_training_allowed: `false`
- current_importance_training_allowed: `false`
- context_utility_claim_allowed: `false`
