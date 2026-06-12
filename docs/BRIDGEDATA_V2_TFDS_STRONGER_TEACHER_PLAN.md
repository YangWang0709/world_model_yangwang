# BridgeData V2 TFDS Stronger Teacher Plan

- recommended_primary_next_step: `Step29 expand token extraction to 32 windows and run train/val context utility sanity`
- recommended_secondary_next_step: `Step30 trained-predictor occlusion teacher`
- do_not_train_current_importance_yet: `true`

## Candidate Plans

### Candidate A: stronger trained predictor occlusion teacher

Train a small predictor on more BridgeData token samples, then compute context-token occlusion delta loss using the trained predictor.

- recommended_stage: `Step30`
- allowed_now: `false`

### Candidate B: train/val context utility experiment first

Expand from 4 windows to 32 or 64 windows, train a tiny predictor on a train split, and evaluate held-out context utility.

- recommended_stage: `Step29`
- allowed_now: `false`

### Candidate C: true temporal VideoMAE token extraction

Replace frame-repeat tokenization with true temporal clip tokens in a later ablation.

- recommended_stage: `after Step29 or Step30`
- allowed_now: `false`

### Candidate D: current importance diagnostic only

Compute current-token occlusion sensitivity as a diagnostic, but do not train a current selector and do not compress current tokens.

- recommended_stage: `diagnostic after context utility is validated`
- allowed_now: `false`

### Candidate E: scale TFDS windows before stronger teacher

Use the existing downloaded shard and Step23 windows to increase from 4 to 32/64 windows.

- recommended_stage: `Step29`
- allowed_now: `false`
