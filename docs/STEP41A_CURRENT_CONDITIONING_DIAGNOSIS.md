# Step41A Current-Conditioning Diagnosis

Step41A is a bounded diagnostic of how the student selector looks at the current observation while choosing historical context tokens.
It is not final selector training, not downstream training, not current importance training, and not evidence of final context utility.

Allowed:
- prepare and run bounded current-conditioning selector smoke training
- update only the lightweight current-conditioned selector head
- keep VideoMAE frozen and unloaded
- use strict shard-aware within, cross, and mixed validation splits
- reuse existing local token and importance artifacts only

Not allowed:
- claim context utility
- train final selector, current importance, world model, VLM, RL, or downstream tasks
- use downstream task improvement as evidence
- download extra datasets, models, images, videos, checkpoints, or archives
- save checkpoint or state_dict artifacts

- variants_run: `["no_current_context_only", "current_mean_summary", "current_frame_summary_attention", "current_coarse_spatial_query_attention", "current_context_similarity_features"]`
- best_variant: `current_context_similarity_features`
- optimizer_scope: `current_conditioned_selector_head_only`
- num_curves: `20`
