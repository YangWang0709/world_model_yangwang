# BridgeData V2 User Subset Checklist

Before running Step22 validation, prepare:

- at least 1 trajectory
- recommended 5-10 trajectories
- at least 24 frames per trajectory
- recommended 40 or more frames per trajectory
- frame names like `frame_000000.jpg`, `frame_000001.jpg`, ...
- `manifest.jsonl` with `trajectory_id` and `num_frames`
- either `image_dir` or `frame_paths` for each manifest row
- optional `actions_path`, `language_instruction`, `goal_image_path`, `task_id`, and `environment_id`
- optional `metadata.json` inside each trajectory directory when using directory autodiscovery

Do not commit the subset directory or any generated data to git.

Forbidden git artifacts include images, videos, archives, `.npy`, `.npz`, `.hdf5`,
`.h5`, `.tfrecord`, `.pt`, checkpoints, model weights, `runs/`, logs, passwords,
and tokens.

Step22 only checks paths and metadata. It does not train, download, extract tokens,
or generate importance shards.
