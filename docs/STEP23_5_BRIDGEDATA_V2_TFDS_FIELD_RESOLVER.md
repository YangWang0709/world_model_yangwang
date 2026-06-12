# STEP23.5 BridgeData V2 TFDS Field Resolver

Step23 validated the official BridgeData V2 TFDS/RLDS mini-shard route and generated
real manifest/window records. It also exposed one Step24 blocker: image candidates
include both metadata flags and true image tensor fields.

The metadata flags look like:

```text
episode_metadata/has_image_0
episode_metadata/has_image_1
episode_metadata/has_image_2
episode_metadata/has_image_3
```

Those fields are booleans or metadata indicators. They are not image tensors and must
not be used by later VideoMAE token extraction.

The real image tensor fields are:

```text
steps/observation/image_0
steps/observation/image_1
steps/observation/image_2
steps/observation/image_3
```

Step23.5 adds a field resolver that rejects metadata flags and resolves the image
field to `steps/observation/image_0`.

## Scope

- fix TFDS/RLDS field selection
- regenerate resolved manifest and window manifest from the Step23 schema summary
- preserve action, language, and goal as metadata only
- keep `env_isaaclab` free of TensorFlow and TFDS

## Not In Scope

- no new TFDS shard download
- no raw zip download
- no full TFDS download
- no training
- no token extraction
- no importance generation
- no VLM/RL/action-conditioned model

## Step24 Readiness

Step24 can use the resolved manifest only when:

- `image_field = steps/observation/image_0`
- `image_field_valid = true`
- `image_field_is_metadata_flag = false`
- `language_field = steps/language_instruction`
- `action_field = steps/action`
- resolved manifest and windows are regenerated
