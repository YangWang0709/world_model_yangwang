# Step34 Proxy-Supervised Temporal Selector Plan

Step34 converts the stable Step33B proxy temporal signal into a safe selector-training scaffold.

Allowed:
- This step may prepare proxy-supervised selector training.
- This step may instantiate a lightweight temporal selector head for dry-run forward checks.
- This step may read existing local Step33B token and proxy-importance artifacts.
- This step must use strict shard-aware splits.

Not allowed:
- This step must not claim context utility.
- This step must not train final selector/current importance.
- This step must not use downstream task improvement as evidence.
- This step must not download extra datasets or models.
- This step must keep VideoMAE frozen.
- This step must not save selector checkpoints.

Step33B evidence used for planning:
- proxy_signal_stable_across_shards: `true`
- within proxy gain/current: `0.15703542878720414`
- cross proxy gain/current: `0.38525779640621804`
- mixed proxy gain/current: `0.3696984868098719`
- dataset_bias_detected: `false`
- full_context_noise_confirmed: `true`

Step34 decision:
- may_prepare_proxy_supervised_selector_training: `true`
- selector_forward_dry_run_performed: `true`
- selector_training_allowed: `false`
- current_importance_training_allowed: `false`
- context_utility_claim_allowed: `false`
- future_selector_training_gate_ready: `false`
