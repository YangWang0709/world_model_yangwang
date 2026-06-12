# BridgeData V2 Step34 Selector Training Gates

Future selector training can only be considered after all gates below are satisfied in a bounded follow-up step.

- selector beats random baseline: `pending`
- selector beats current-only baseline: `pending`
- selector generalizes across shard: `pending`
- no language/trajectory leakage: `pass`
- no dataset bias or shard shift: `pass`
- full-context-noisy issue acknowledged: `pass`

Required evaluation design:
- within-shard validation: shard1 train/validation split with disjoint sample IDs and trajectories where possible.
- cross-shard validation: train shard0 -> validate shard1 and train shard1 -> validate shard0.
- mixed-shard validation: shard-aware mixed training and validation with per-shard breakdown.

Decision flags remain conservative:
- selector_training_allowed: `false`
- current_importance_training_allowed: `false`
- context_utility_claim_allowed: `false`
