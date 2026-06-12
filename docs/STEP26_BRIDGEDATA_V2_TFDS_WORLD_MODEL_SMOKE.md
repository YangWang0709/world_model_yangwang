# Step26 BridgeData V2 TFDS World-Model Smoke

Step26 follows the two BridgeData TFDS smoke stages that already succeeded:

- Step24: real TFDS image windows were encoded into local-only VideoMAE token artifacts.
- Step25: those token artifacts received context-token proxy importance labels.

This step validates the next dataflow edge:

```text
Step24 tokens + Step25 importance
-> context bottleneck topK selection
-> full current tokens + selected context memory
-> world-model forward/loss smoke
```

It is not formal training. It does not run an optimizer step, does not train a
teacher, selector, or world model, and does not download data or models. The
smoke predictor is allowed to be randomly initialized only to verify tensor
plumbing and finite losses. Its loss table is not final model performance.

The policies evaluated are:

- `current_only`
- `random_context_topk`
- `proxy_importance_topk`
- `full_context_reference`

The current tokens remain full for every policy. Only context tokens are
selected or dropped.
