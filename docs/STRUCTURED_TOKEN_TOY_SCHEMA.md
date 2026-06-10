# Structured Token Toy Schema

The `structured_toy` split is a small deterministic diagnostic dataset for Step 5.5. It does not download real data or use a large encoder. It creates token shards directly with the existing Step 3 token shard schema.

Each sample contains:

- `past_tokens`: `[N, D]`
- `future_tokens`: `[N, D]`
- `key_token_indices`: stored in per-sample `metadata`
- `key_token_mask`: stored in optional `aux_labels.key_token_mask` as `[B, N]`
- `sample_id`
- `task_text`
- `metadata`

Default generation settings:

- `num_samples`: 32
- `num_tokens`: 196
- `token_dim`: 768
- `num_key_tokens`: 4
- `split`: `structured_toy`
- `signal_scale`: 3.0
- `noise_scale`: 0.05
- `seed`: 42

The future target is intentionally task-grounded to a small set of key past tokens. The generator injects a sample-specific signal into the selected key tokens and writes the same global signal into `future_tokens`, so `future_tokens.mean(dim=1)` depends on the selected key tokens. Teacher occlusion never reads `key_token_indices`; those labels are only used by quality evaluation.

Generated `.pt` shards live under `data/token_shards/structured_toy/` and must remain untracked.
