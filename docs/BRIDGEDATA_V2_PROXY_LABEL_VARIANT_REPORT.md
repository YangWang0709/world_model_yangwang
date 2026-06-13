# BridgeData V2 Proxy Label Variant Report

- num_samples: `128`
- num_variants: `13`
- global_spatial_prior_stats: `{"max": 1.0, "mean": 0.2537931799888611, "min": 0.0, "shape": [392], "std": 0.11496634781360626, "top256_mass_ratio": 0.8063830766222182, "top64_mass_ratio": 0.28193212013849467}`

Top diagnosis rows:
- `{"cross_shard_consistent": true, "cross_shard_cosine": 0.9858646988868713, "diagnosis_score": 0.5926670813463912, "per_frame_spatial_entropy_mean_mean": 0.9855158105492592, "reconstruction_mse_to_original_mean": 0.013967427767056506, "residual_energy_after_variant_mean": 0.1671388539322634, "score_is_engineering_heuristic": true, "top256_mass_ratio_mean": 0.15204364838619966, "variant_name": "global_spatial_prior_removed_residual"}`
- `{"cross_shard_consistent": true, "cross_shard_cosine": 0.9956435561180115, "diagnosis_score": 0.5741292233451964, "per_frame_spatial_entropy_mean_mean": 0.9999999953433871, "reconstruction_mse_to_original_mean": 0.02516671532066539, "residual_energy_after_variant_mean": 1.4920549541881225e-14, "score_is_engineering_heuristic": true, "top256_mass_ratio_mean": 0.08791730042326396, "variant_name": "temporal_broadcast"}`
- `{"cross_shard_consistent": true, "cross_shard_cosine": 0.9956435561180115, "diagnosis_score": 0.5741292233451964, "per_frame_spatial_entropy_mean_mean": 0.9999999953433871, "reconstruction_mse_to_original_mean": 0.02516671532066539, "residual_energy_after_variant_mean": 1.4920549541881225e-14, "score_is_engineering_heuristic": true, "top256_mass_ratio_mean": 0.08791730042326396, "variant_name": "temporal_only"}`
- `{"cross_shard_consistent": true, "cross_shard_cosine": 0.8451917171478271, "diagnosis_score": 0.5730542818105759, "per_frame_spatial_entropy_mean_mean": 0.8269504550844431, "reconstruction_mse_to_original_mean": 0.020562993580824696, "residual_energy_after_variant_mean": 0.7873217179923387, "score_is_engineering_heuristic": true, "top256_mass_ratio_mean": 0.4305818603913265, "variant_name": "denoised_soft_topk_512"}`
- `{"cross_shard_consistent": true, "cross_shard_cosine": 0.9434122443199158, "diagnosis_score": 0.5568640947722665, "per_frame_spatial_entropy_mean_mean": 0.8702281005680561, "reconstruction_mse_to_original_mean": 0.007863453811296495, "residual_energy_after_variant_mean": 0.4652236782439569, "score_is_engineering_heuristic": true, "top256_mass_ratio_mean": 0.18784028043652454, "variant_name": "coarse_index_bins_196"}`
- `{"cross_shard_consistent": true, "cross_shard_cosine": 0.838678240776062, "diagnosis_score": 0.5506549793593877, "per_frame_spatial_entropy_mean_mean": 0.8578117084689438, "reconstruction_mse_to_original_mean": 0.0268322675183299, "residual_energy_after_variant_mean": 0.8354261244014619, "score_is_engineering_heuristic": true, "top256_mass_ratio_mean": 0.585073294491724, "variant_name": "denoised_soft_topk_256"}`

The score is an engineering diagnosis heuristic, not a scientific proof that a selector is ready.
