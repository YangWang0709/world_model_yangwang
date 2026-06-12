# BridgeData V2 TFDS to Long-Context Mapping

Step23 maps RLDS episodes into the existing long-context data interface without
creating a new schema.

## RLDS to Manifest

- RLDS episode -> trajectory
- RLDS step sequence -> frame index sequence
- observation image field -> future token extraction source metadata
- action/proprio/language/goal -> metadata only
- valid trajectory -> `num_frames >= 24`

Manifest records follow `data/bridgedata_v2_manifest_schema.py`:

```json
{
  "dataset_name": "BridgeData V2",
  "split": "train",
  "trajectory_id": "tfds_episode_000001",
  "num_frames": 38,
  "frame_paths": [],
  "image_dir": null,
  "camera_names": ["steps/observation/image_0"],
  "actions_path": null,
  "actions": null,
  "language_instruction": null,
  "goal_image_path": null,
  "task_id": null,
  "environment_id": null,
  "metadata": {
    "source": "tfds_rlds_mini_shard",
    "tfds_episode_index": 1,
    "image_field": "steps/observation/image_0",
    "action_field": "steps/action",
    "language_field": "steps/language_instruction",
    "goal_field": null,
    "use_action_as_input": false,
    "use_language_as_input": false,
    "use_goal_image_as_input": false,
    "save_video_tensors": false
  }
}
```

## Window Builder

Step23 reuses `build_bridgedata_windows_from_manifest()` from Step20 with:

```text
context_len=16
current_len=4
future_len=4
stride=1
```

No video tensors are saved. The window manifest records only frame indices and
metadata needed for a later bounded TFDS token extraction dry-run.

## Metadata Boundary

Action, proprioception, language, and goal fields are not model inputs in Step23.
They are preserved so later analysis can decide what to do with them under a separate
approval gate.
