# BridgeData V2 Current-Conditioning Variant Report

All metrics in this report are recomputed against the Step41A redesigned target label.
Step40A metrics are background only and are not reused as this round's result.

- no_current_mse: `0.018570232205092907`
- current_mean_summary_mse: `0.018887842074036598`
- best_variant_within_mse: `0.018689891323447227`
- best_variant_cross_mse: `0.017623314633965492`
- best_variant_mixed_mse: `0.018377697095274925`
- best_variant_beats_no_current: `false`
- best_variant_beats_current_mean_summary: `false`
- beats_random: `true`
- beats_uniform_or_mean: `false`
- beats_temporal_broadcast: `false`
- beats_train_global_spatial_prior: `true`
- cross_shard_generalization: `false`
- top256_overlap_mean: `0.1209582967018329`
- overfit_gap_mean_abs: `0.00856961046035091`

The coarse spatial query variant uses contiguous token-index bins. This is a low-cost diagnostic heuristic, not true pixel geometry.

Ranked variants: `[{"current_gain_over_no_current": 0.0004916777834296227, "gain_over_current_mean_summary": 0.0008092876523733139, "num_rows": 4, "selector_val_mse": 0.018078554421663284, "top256_overlap": 0.1214779436740729, "variant_name": "current_context_similarity_features"}, {"current_gain_over_no_current": 0.0004642922431230545, "gain_over_current_mean_summary": 0.0007819021120667458, "num_rows": 4, "selector_val_mse": 0.018105939961969852, "top256_overlap": 0.10545838495044757, "variant_name": "current_frame_summary_attention"}, {"current_gain_over_no_current": 0.0, "gain_over_current_mean_summary": 0.00031760986894369125, "num_rows": 4, "selector_val_mse": 0.018570232205092907, "top256_overlap": 0.11889954393781968, "variant_name": "no_current_context_only"}, {"current_gain_over_no_current": -0.00031760986894369125, "gain_over_current_mean_summary": 0.0, "num_rows": 4, "selector_val_mse": 0.018887842074036598, "top256_overlap": 0.107772038842711, "variant_name": "current_mean_summary"}, {"current_gain_over_no_current": -0.0028489427641034126, "gain_over_current_mean_summary": -0.0025313328951597214, "num_rows": 4, "selector_val_mse": 0.02141917496919632, "top256_overlap": 0.10645686391064578, "variant_name": "current_coarse_spatial_query_attention"}]`
