# Step29 BridgeData V2 TFDS Train-Val Context Utility

Step29 follows the Step28 finding that 4-sample overfit is memorization-prone.

- uses existing TFDS shard only
- expands to a bounded 32-window train/val sanity run
- performs limited VideoMAE token extraction and proxy importance generation
- trains only the tiny world-model predictor
- does not train selector or current importance
- does not claim final context utility

- pass: `true`
- context_utility_sanity_signal: `inconclusive`
