# Step30B BridgeData V2 TFDS Occlusion Teacher

Step30B follows the Step30A finding that proxy topK is stable while full context remains noisy.

- trains only a small current-conditioned predictor teacher
- freezes the teacher and generates context-token occlusion delta labels
- compares teacher labels to Step30A proxy labels
- optionally runs teacher_topK tiny utility sanity
- does not train selector or current importance
- does not claim final context utility

- pass: `true`
- teacher_beats_proxy_fraction: `0.0`
