# BridgeData V2 User Subset Template

Prepare a tiny local subset under:

```text
data/bridgedata_v2_tiny_user_subset/
  manifest.jsonl
  traj_000001/
    images/
      frame_000000.jpg
      frame_000001.jpg
      ...
    metadata.json
    actions.npy or actions.json optional
    goal.jpg optional
```

## Manifest JSONL

Each line should describe one trajectory:

```json
{
  "trajectory_id": "traj_000001",
  "split": "train",
  "num_frames": 40,
  "image_dir": "traj_000001/images",
  "camera_names": ["main"],
  "actions_path": "traj_000001/actions.npy",
  "language_instruction": "put the object in the bowl",
  "goal_image_path": "traj_000001/goal.jpg",
  "task_id": "put_object_in_bowl",
  "environment_id": "kitchen_01",
  "metadata": {
    "source": "user_provided_tiny_subset"
  }
}
```

`trajectory_id` and `num_frames` are required. Provide either `frame_paths` or
`image_dir`. Relative paths are resolved against `data/bridgedata_v2_tiny_user_subset/`.

## Directory Autodiscovery

If `manifest.jsonl` is absent, Step22 can scan trajectory directories:

```text
traj_000001/
  images/ or frames/
  metadata.json optional
  actions.npy or actions.json optional
  goal.jpg optional
```

`metadata.json` may include:

```json
{
  "trajectory_id": "traj_000001",
  "split": "train",
  "language_instruction": "put the object in the bowl",
  "task_id": "put_object_in_bowl",
  "environment_id": "kitchen_01",
  "camera_names": ["main"],
  "actions_path": "traj_000001/actions.npy",
  "goal_image_path": "traj_000001/goal.jpg",
  "metadata": {
    "source": "user_provided_tiny_subset"
  }
}
```

## Metadata Policy

Action, language, camera, and goal-image fields are preserved as metadata only.
They are not used as model inputs in Step22.
