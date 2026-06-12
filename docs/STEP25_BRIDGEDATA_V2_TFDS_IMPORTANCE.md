# Step25 BridgeData V2 TFDS Importance Dry-Run

Step25 follows the Step24 BridgeData V2 TFDS mini-shard token extraction dry-run.
Step24 proved that real TFDS images can be exported into local clip caches and
encoded into local-only VideoMAE token artifacts.

This step only validates the next pipeline edge:

```text
Step24 token artifacts -> context-token importance labels -> importance manifest
```

It is not a training stage, not full importance generation, and not a selector or
world-model stage. It reads the four Step24 token smoke artifacts, validates the
expected token shapes, and writes small smoke artifacts under
`runs/bridgedata_v2_tfds_importance_step25_v1/`.

The method is `proxy_token_mse_dryrun`. It is a deterministic token-space proxy
for pipeline validation only:

- context tokens are scored with shape `[16, 392]`.
- temporal aggregation has shape `[16]`.
- spatial aggregation has shape `[392]`.
- current tokens are kept full.
- current-token importance is not generated.
- action, language, and goal metadata are not used as model inputs.

The output labels are not final teacher labels and should not be described as
scientific results.
