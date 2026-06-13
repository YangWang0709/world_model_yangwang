# Step38B Proxy Label Learnability Diagnosis

Step38B may diagnose and prepare evidence for proxy-supervised selector work.
Step38B must not claim context utility.
Step38B must not train final selector, current importance, VideoMAE, V-JEPA, VLM, world model, downstream model, RL, or action-conditioned model.
Step38B must not use downstream task improvement as evidence.
Step38B must not download extra datasets or models.
Step38B keeps VideoMAE frozen by not loading or updating it.
Step38B uses strict shard-aware splits only for leakage checks and cross-shard diagnosis.

Questions:
1. Does the temporal component explain most variance? `not established`.
2. Is the spatial residual stable, concentrated, and cross-shard consistent? `no, not enough evidence`.
3. Is the current patch-level label suitable for continued patch selector training? `no, redesign or denoise the proxy label first`.
4. Recommended Step39: `label redesign toward temporal-only or temporal+broadcast/coarse supervision`.
5. This step performed no selector/current-importance/final-selector/world-model training and no new data/model downloads.

Key metrics:
- safe_stop: `false`
- num_samples: `128`
- residual_energy_ratio_mean: `0.5584554279577573`
- temporal_broadcast_r2_like_mean: `0.1723712056739702`
- spatial_entropy_mean: `0.8236089670099318`
- cross_shard_residual_cosine: `0.9777631163597107`
