# BridgeData V2 User-Provided Subset Instructions

If Step21 safe-stops, prepare a tiny local subset like this:

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
  traj_000002/
    ...
```

Requirements:

- at least 1 trajectory
- recommended 5-10 trajectories
- each trajectory needs at least 24 frames
- 40+ frames per trajectory is preferred
- manifest.jsonl must include trajectory_id and num_frames
- provide either frame_paths or image_dir
- language_instruction, goal_image_path, and actions_path are optional metadata
- do not commit this directory to git
