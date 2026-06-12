# BridgeData V2 TFDS Field Selection Policy

## Image Fields

Image fields must satisfy all of these rules:

- start with `steps/observation/`
- contain `image`
- not contain `episode_metadata`
- not be a `has_image_*` flag

Preferred order:

```text
steps/observation/image_0
steps/observation/image_1
steps/observation/image_2
steps/observation/image_3
```

If only `episode_metadata/has_image_*` fields exist, the resolver blocks manifest
generation instead of producing a misleading valid manifest.

## Action Fields

Preferred action field:

```text
steps/action
```

Action is stored as metadata only and is never used as model input in Step23.5.

## Language Fields

Preferred language order:

```text
steps/language_instruction
steps/natural_language_instruction
steps/language_embedding
```

Text instructions are preferred over embeddings. If only an embedding exists, the
resolver may select it as metadata but records a warning.

## Goal Fields

Preferred goal order:

```text
steps/observation/goal_image
steps/goal_image
steps/goal
```

BridgeData V2 mini-shard Step23 did not expose a goal field, so `goal_field = null` is
allowed and is not blocking.

## Manifest Metadata

Resolved manifest records include:

```json
{
  "image_field": "steps/observation/image_0",
  "image_field_valid": true,
  "image_field_is_metadata_flag": false,
  "action_field": "steps/action",
  "use_action_as_input": false,
  "language_field": "steps/language_instruction",
  "use_language_as_input": false,
  "goal_field": null,
  "use_goal_image_as_input": false
}
```

The manifest does not export images, video tensors, action arrays, tokens, or
importance values.
