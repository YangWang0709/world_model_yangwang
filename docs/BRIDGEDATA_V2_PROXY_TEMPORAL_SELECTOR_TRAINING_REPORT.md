# BridgeData V2 Proxy Temporal Selector Training Report

- num_curves: `12`
- within_shard: `{"current_only_baseline_mse": 0.09639798601468404, "num_rows": 3, "overfit_gap": 0.04866924633582433, "random_baseline_mse": 0.20968981583913168, "selector_beats_current_only_baseline": false, "selector_beats_random_baseline": true, "selector_beats_uniform_baseline": true, "selector_pearson": 0.2236176216404191, "selector_spearman": 0.2581242779543733, "selector_val_mse": 0.11556324114402135, "top1_frame_hit": 0.1593137254901961, "top2_frame_overlap": 0.2218137254901961, "top4_frame_overlap": 0.43106617647058826, "uniform_baseline_mse": 0.19560803472995758}`
- cross_shard: `{"current_only_baseline_mse": 0.09334258983532588, "num_rows": 6, "overfit_gap": 0.06641046081980069, "random_baseline_mse": 0.20564383268356323, "selector_beats_current_only_baseline": false, "selector_beats_random_baseline": true, "selector_beats_uniform_baseline": true, "selector_pearson": 0.04864910953705467, "selector_spearman": 0.005676923021810096, "selector_val_mse": 0.13219810898105303, "top1_frame_hit": 0.14910676004447035, "top2_frame_overlap": 0.24244880859793702, "top4_frame_overlap": 0.3041972883419831, "uniform_baseline_mse": 0.2005630706747373}`
- mixed_shard: `{"current_only_baseline_mse": 0.0930618221561114, "num_rows": 3, "overfit_gap": 0.03995764752229055, "random_baseline_mse": 0.20004275937875113, "selector_beats_current_only_baseline": false, "selector_beats_random_baseline": true, "selector_beats_uniform_baseline": true, "selector_pearson": -0.008887046774724077, "selector_spearman": -0.01156286011066507, "selector_val_mse": 0.11392403145631154, "top1_frame_hit": 0.11897759103641457, "top2_frame_overlap": 0.147093837535014, "top4_frame_overlap": 0.264828431372549, "uniform_baseline_mse": 0.2012647440036138}`
- selector_beats_random_baseline: `true`
- selector_beats_current_only_baseline: `false`
- selector_generalizes_cross_shard: `false`

The selector input is limited to existing context/current token summaries. Action and language are not model inputs.
