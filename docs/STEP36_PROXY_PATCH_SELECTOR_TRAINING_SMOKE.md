# Step36 Proxy Patch/Token Selector Training Smoke

Step36 trains only a bounded patch/token-level selector head smoke.
Step36 does not train the final selector.
Step36 does not train current importance.
Step36 does not claim context utility.
Step36 does not use downstream task improvement as evidence.

Allowed:
- Read existing Step33B token artifacts.
- Read existing Step33B proxy importance artifacts.
- Read Step35 temporal selector outputs as baseline reference.
- Train only `ProxyPatchTokenSelectorHead`.

Not allowed:
- No raw zip, full TFDS, DROID, images, videos, checkpoints, or new model downloads.
- No VideoMAE, world model, downstream predictor, policy, VLM/RL, or action-conditioned training.
- No action or language as model input.
- No writes to `data/token_shards/` or `data/importance_shards/`.

Safety flags:
- optimizer_scope: `proxy_patch_token_selector_head_only`
- checkpoint_saved: `false`
- selector_training_allowed: `false`
- final_selector_training_allowed: `false`
- current_importance_training_allowed: `false`
- context_utility_claim_allowed: `false`
