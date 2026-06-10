# Token Shard Schema

Step 3 saves offline token shards so later teacher and student experiments can reuse frozen encoder outputs without repeatedly running video encoders.

The shard reader is designed so future V-JEPA or VideoMAE integration can replace only the encoder implementation while preserving the same shard format.

## Schema Version

Current schema version: `0.1.0`

## File Format

Each token shard is a `.pt` file saved with `torch.save`. The payload must be a Python dict.

Required keys:

- `schema_version`: schema version string, currently `0.1.0`.
- `encoder_name`: encoder identifier, for Step 3 `dummy_video_encoder`.
- `encoder_config`: encoder metadata such as token count, token dim, and patch grid.
- `created_at`: ISO timestamp.
- `split`: dataset split, for Step 3 `toy`.
- `sample_ids`: list of sample ids, length `B`.
- `task_texts`: list of task strings, length `B`.
- `past_tokens`: tensor with shape `[B, N, D]`.
- `future_tokens`: tensor with shape `[B, N, D]` or `[B, D]`.
- `metadata`: list of metadata dicts, length `B`.

## Shape Conventions

For the Step 3 dummy encoder:

- `B`: shard batch size.
- `N`: number of visual tokens, default `196`.
- `D`: token dimension, default `768`.

The default Step 3 toy extraction produces:

- `past_tokens`: `[8, 196, 768]`
- `future_tokens`: `[8, 196, 768]`

## Generated Artifacts

Token shards under `data/token_shards/` are generated artifacts and should not be committed to git.

Toy `.pt` clips under `data/toy_videos/` are generated artifacts and should not be committed to git.

