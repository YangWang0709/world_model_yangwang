# Step30A BridgeData V2 TFDS Window Diversity

Step30A follows the Step29 inconclusive train/val context utility sanity result.

- uses the existing BridgeData V2 TFDS shard only
- expands selected windows and trajectory coverage inside the existing Step23.5 manifest
- runs multi-seed trajectory-disjoint train/val splits
- performs limited token extraction and proxy importance generation
- trains only the tiny world-model predictor
- does not train selector or current importance
- does not claim final context utility

- pass: `true`
- context_signal_stability: `weak_proxy_positive_full_context_noisy`
