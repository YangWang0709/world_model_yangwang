# BridgeData V2 TFDS Field Resolver Report

Field resolver patch passed.

## Resolved Fields

- image_field: `steps/observation/image_0`
- image_field_valid: `true`
- image_field_is_metadata_flag: `false`
- rejected_metadata_image_flags: `['episode_metadata/has_image_0', 'episode_metadata/has_image_1', 'episode_metadata/has_image_2', 'episode_metadata/has_image_3']`
- action_field: `steps/action`
- action_used_as_input: `false`
- language_field: `steps/language_instruction`
- language_used_as_input: `false`
- goal_field: `None`
- goal_used_as_input: `false`

## Regenerated Outputs

- num_manifest_records: `10`
- num_windows: `155`
- resolved_manifest_path: `/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_field_resolver_step23_5_v1/tfds_mini_manifest_resolved.jsonl`
- resolved_window_manifest_path: `/home/ubuntu22/tgpawb_world_model/runs/bridgedata_v2_tfds_field_resolver_step23_5_v1/tfds_mini_window_manifest_resolved.jsonl`

## Boundaries

- no download
- no training
- no token extraction
- no importance generation
- action/language/goal remain metadata only

## Recommended Step24

- name: `BridgeData V2 TFDS mini-shard token extraction dry-run`
- condition: `resolved image field is steps/observation/image_0 and manifest/window regenerated`
- no_training: `true`
