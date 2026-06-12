# Step28 BridgeData V2 TFDS Context Sanity

Step28 is a read-only diagnosis and decision step after the Step27 tiny overfit run.

## Why Step28 Exists

- Step27 proved that a tiny predictor can overfit 4 real BridgeData token samples.
- Step27 did not prove final model performance or context bottleneck utility.
- The current_only policy also overfit, so 4 samples are too small to support a context advantage claim.
- Step28 keeps current tokens full and does not train current importance.

## Guardrail Result

- pass: `true`
- safe_stop: `false`
- context_utility_claim_allowed: `false`
- train_current_importance_now: `false`
