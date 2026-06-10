# Importance Shard Schema

Step 5 generates predictive token importance labels for later student selector training.
For each visual token, the trained full-token teacher is evaluated once with that token masked.
The label is the increase in future latent prediction loss caused by the occlusion:

```text
I_i = Loss(Teacher(z_without_i), z_future) - Loss(Teacher(z_full), z_future)
```

Large positive values mean the token was useful for the teacher's future prediction.
Negative values are preserved by default because masking can occasionally reduce loss on a tiny toy setup.

## File Format

Importance shards are PyTorch `.pt` files containing a dictionary with schema version `0.1.0`.
Generated shards live under ignored directories such as `data/importance_shards/` and are not committed.

Required keys:

```text
schema_version
importance_method
teacher_checkpoint
teacher_config
source_token_shard
created_at
split
sample_ids
task_texts
importance_scores
importance_scores_norm
base_losses
masked_losses
metadata
mask_config
```

## Tensor Shapes

```text
importance_scores: [B, N]
importance_scores_norm: [B, N]
base_losses: [B]
masked_losses: [B, N]
```

`B` is the number of samples in the source token shard. `N` is the number of visual tokens per sample.
For the dummy Step 3 shards, the expected first-shard shape is `[8, 196]`.

## Key Meanings

`importance_scores` stores raw loss deltas. `importance_scores_norm` stores min-max normalized scores per sample.
`base_losses` stores the full-token teacher loss per sample. `masked_losses` stores one loss per masked token.
`teacher_checkpoint` and `teacher_config` identify the teacher used to produce the labels.
`source_token_shard` points back to the token shard used as input.

## Later Student Use

The Step 6 selector can train its token scores against `importance_scores` or `importance_scores_norm`.
A first version can supervise top-K or soft-mask selection on the dummy token shards before moving to real frozen video encoder tokens.
