# BridgeData V2 TFDS Token Extraction Dataflow

Step24 uses a two-process dataflow to avoid polluting the main VideoMAE environment.

1. The TFDS environment reads the already downloaded local BridgeData TFDS mini shard.
2. The exporter selects Step23.5 resolved windows and reads `steps/observation/image_0`.
3. It writes tiny `.npz` clip caches containing `context_video`, `current_video`, `future_video`, and metadata only.
4. The `env_isaaclab` environment reads those `.npz` files without importing TensorFlow or TFDS.
5. The token script uses the existing local-only VideoMAE wrapper and writes small token smoke artifacts under the Step24 run directory.
6. The manifest records context/current/future token shapes, image field, and false input-use flags for action, language, and goal.

No image files or video files are emitted. Action, language, and goal fields are metadata only; they are not model inputs in this stage.
