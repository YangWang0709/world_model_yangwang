# BridgeData V2 Proxy Label Decomposition Report

The label decomposition is statistical only:
- proxy label: `[16,392]`
- temporal component: `[16]`
- temporal broadcast: `[16,392]`
- spatial residual: `[16,392]`

- num_rows: `128`
- all_samples: `{"num_rows": 128, "per_frame_spatial_entropy_mean_mean": 0.8236089670099318, "residual_energy_ratio_mean": 0.5584554279577573, "spatial_residual_learnable_evidence": false, "temporal_broadcast_r2_like_mean": 0.1723712056739702, "temporal_component_dominates": false, "temporal_energy_ratio_mean": 0.44154457920166335, "top256_mass_ratio_mean": 0.2145879373130259, "top256_residual_mass_ratio_mean": 0.14874840245564575}`
- cross_shard: `{"computed": true, "cross_shard_residual_consistent": true, "cross_shard_residual_cosine": 0.9777631163597107, "cross_shard_residual_l1": 2.6524998247623444e-05, "num_samples_by_shard": {"shard0": 64, "shard1": 64}, "shards": ["shard0", "shard1"]}`

Action, language, future tokens, current importance, and downstream labels are not used as diagnosis inputs.
