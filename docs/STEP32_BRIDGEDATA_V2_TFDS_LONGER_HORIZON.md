# Step32 BridgeData V2 TFDS Longer-Horizon Diagnosis

Step32 follows Step31, where the delta target exposed a stronger context-gain signal than plain future summaries.

- rebuilds longer-horizon windows from the existing BridgeData V2 TFDS mini shard
- keeps context/current/future lengths fixed at 16/4/4
- tests horizon gaps 0/4/8 and optional 12
- performs limited clip export, frame-repeat VideoMAE token extraction, and proxy importance generation
- trains only tiny diagnostic predictors
- does not train selector or current importance
- does not claim final context utility

- pass: `true`
- horizons_completed: `[0, 4, 8, 12]`
- best_target_variant: `future_delta_last_minus_current`
