# Step24 BridgeData V2 TFDS Token Extraction Dry-Run

Step24 validates the next narrow link after Step23 and Step23.5: real BridgeData V2 TFDS/RLDS image frames are exported through the resolved image tensor field and passed into the existing local-only VideoMAE encoder path.

This stage is intentionally small. It uses the Step23.5 resolved manifest and window manifest, selects only a few windows, writes temporary `.npz` clip caches under the Step24 run directory, and writes small `.pt` token smoke artifacts under the same ignored run directory.

The required image field is `steps/observation/image_0`. Metadata flags such as `episode_metadata/has_image_0` are rejected.

Step24 does not train VideoMAE, V-JEPA, a teacher, a selector, or a world model. It does not generate predictive importance labels. It does not perform full token extraction, does not write `data/token_shards/`, and does not write `data/importance_shards/`.

The TFDS/RLDS read is isolated to `/home/ubuntu22/miniconda3/envs/tgpawb_tfds_py311/bin/python`. The VideoMAE token pass runs in `env_isaaclab`, which must remain free of TensorFlow and TFDS.
