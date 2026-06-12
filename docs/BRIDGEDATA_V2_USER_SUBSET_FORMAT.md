# BridgeData V2 User Subset Format

Put user-provided tiny subsets under an ignored local-only directory:

```text
data/bridgedata_v2_tiny_user_subset/
  manifest.jsonl
  traj_000001/
    images/
      frame_000000.jpg
      frame_000001.jpg
    metadata.json
    actions.npy or actions.json
    goal.jpg optional
```

Each `manifest.jsonl` line should include:

- trajectory_id
- split
- num_frames
- frame_paths or image_dir
- camera_names
- actions_path
- language_instruction
- goal_image_path
- task_id
- environment_id
- metadata

The subset directory is ignored by git. Step20 only reads manifest metadata and file listings; it does not read image bytes or action arrays.
