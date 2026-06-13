# Step37 Factorized Selector Diagnosis

Step37 exists because Step36 direct patch/token prediction beat random and uniform baselines, but did not beat temporal broadcast, current-only, or cross-package gates.

Step37 trains only bounded factorized selector heads for diagnosis.
Step37 does not train the final selector.
Step37 does not train current importance.
Step37 does not train VideoMAE.
Step37 does not claim context utility.
Step37 does not use downstream task improvement as evidence.

Design:
- temporal branch predicts frame scores `[16]`.
- spatial residual branch predicts token residuals `[16,392]`.
- combined score is temporal score plus spatial residual.
- proxy temporal prior is used only as an auxiliary training target, not as model input.

Safety:
- optimizer_scope: `proxy_factorized_selector_head_only`
- checkpoint_saved: `false`
- selector_training_allowed: `false`
- current_importance_training_allowed: `false`
- context_utility_claim_allowed: `false`
- no writes to `data/token_shards/` or `data/importance_shards/`.
