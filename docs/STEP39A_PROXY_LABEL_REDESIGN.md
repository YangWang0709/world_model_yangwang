# Step39A Proxy Label Redesign

Step39A changes the proxy supervision target before training another selector.
It is not selector training, because Step35, Step36, Step37, and Step38B showed that the old patch-level answer sheet was too noisy or not learnable enough against stronger baselines.

Compared label variants:
`coarse_grid_7x14_if_14x28`, `coarse_grid_7x7_if_14x28`, `coarse_index_bins_196`, `coarse_index_bins_49`, `coarse_index_bins_98`, `denoised_soft_topk_256`, `denoised_soft_topk_512`, `global_spatial_prior_removed_residual`, `temporal_broadcast`, `temporal_broadcast_minmax`, `temporal_broadcast_rank_soft`, `temporal_only`, `temporal_plus_global_spatial_prior`

Variant meanings:
- `temporal_only`: frame-level `[16]` target with a broadcast copy only for metric comparison.
- `temporal_broadcast`: each frame score is copied to all 392 spatial tokens.
- `coarse_index_bins_*` and `coarse_grid_*`: coarse token groups; grid variants are token-grid heuristics, not real pixel coordinates.
- `denoised_soft_topk_*`: soft topK emphasis with a floor, not a hard one-hot mask.
- global-prior variants test whether stable spatial bias explains the label.

Recommended Step40 label variant: `global_spatial_prior_removed_residual`.
Reason: `highest engineering diagnosis score among safer redesigned labels`.

Safety:
- trained selector/current importance/final selector/world model: `false`
- downloaded new data/model: `false`
- used action/language/future tokens as input: `false`
- final context utility claim: `false`
