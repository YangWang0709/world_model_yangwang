# BridgeData V2 TFDS Mini-Shard Report

TFDS/RLDS mini-shard smoke succeeded.
Real TFDS episodes were inspected.
Manifest and window manifest were generated.
Next step: TFDS mini-shard token extraction dry-run.

## Summary

- safety_gate_pass: `true`
- real_tfds_validated: `true`
- safe_stop: `false`
- download_performed: `true`
- download_bytes: `108416200`
- num_shards_downloaded: `1`
- num_episodes_scanned: `50`
- episode length min/mean/max: `18/40.54/89`
- num_manifest_records: `10`
- num_valid_trajectories: `10`
- num_windows: `155`

## Candidate Fields

- image fields: `['episode_metadata/has_image_0', 'episode_metadata/has_image_1', 'episode_metadata/has_image_2', 'episode_metadata/has_image_3', 'steps/observation/image_0', 'steps/observation/image_1', 'steps/observation/image_2', 'steps/observation/image_3']`
- action fields: `['steps/action']`
- language fields: `['episode_metadata/has_language', 'steps/language_embedding', 'steps/language_instruction']`
- goal fields: `[]`

## Boundaries

- no raw zip download
- no full TFDS download
- no DROID download
- no training
- no token extraction
- no importance generation
- action, language, and goal remain metadata only

## Recommended Step24

- name: `BridgeData V2 TFDS mini-shard token extraction dry-run`
- condition: `real TFDS mini-shard validated`
- no_training: `true`
