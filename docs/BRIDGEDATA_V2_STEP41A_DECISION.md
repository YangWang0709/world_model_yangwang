# BridgeData V2 Step41A Decision

Step40A failed because it beat random and train-global-spatial-prior baselines but did not beat uniform/mean or temporal-broadcast baselines, and cross-shard generalization was false.
Step41A therefore asks whether stronger current-conditioning actually helps the selector choose history, rather than treating this as another final selector attempt.

- current_conditioning_candidate_ready: `false`
- best_variant: `current_context_similarity_features`
- best_variant_beats_no_current: `false`
- best_variant_beats_current_mean_summary: `false`
- best_variant_generalizes_cross_shard: `false`
- reason: `one or more current-conditioning diagnostic gates did not pass`

Recommended Step42:
- name: `Step42 encoder inventory or stronger representation / label rethink`
- scope: `no downstream selector use`

Safety decision flags:
- downstream_selector_use_allowed: `false`
- final_selector_training_allowed: `false`
- current_importance_training_allowed: `false`
- context_utility_claim_allowed: `false`
- checkpoint_saved: `false`
- state_dict_saved: `false`
- new_dataset_download_performed: `false`
- model_download_performed: `false`
- action/language/future tokens used as selector input: `false`
