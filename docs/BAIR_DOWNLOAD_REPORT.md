# BAIR Download Report

- download_success: `true`
- blocking_reason: `None`
- data_dir: `/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_tfds`
- disk_free_before_gib: `189.03737258911133`
- disk_free_after_gib: `189.03736877441406`
- elapsed_time_sec: `0.995246410369873`

```json
{
  "attempts": 1,
  "blocking_reason": null,
  "capability_summary": {
    "bair_data_dir_exists": true,
    "blocking_reason": null,
    "can_attempt_download": true,
    "current_conda_env": "tgpawb_tfds_py311",
    "data_dir": "/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_tfds",
    "disk_free_gib": 189.03737258911133,
    "download_allowed": true,
    "env_isolation": {
      "tensorflow_expected_in_training_env": false,
      "training_env_not_modified": true,
      "uses_isolated_tfds_env": true
    },
    "min_disk_required_gib": 80.0,
    "numpy_available": true,
    "partial_tfds_download_detected": false,
    "proxy_env": {},
    "python_executable": "/home/ubuntu22/miniconda3/envs/tgpawb_tfds_py311/bin/python",
    "python_version": "3.11.15",
    "ram_available_gib": 25.289756774902344,
    "tensorflow_available": true,
    "tensorflow_datasets_available": true,
    "tensorflow_datasets_version": "4.9.10",
    "tensorflow_version": "2.21.0",
    "tfds_builder_available": true,
    "tfds_builder_error": null,
    "tfds_env": "tgpawb_tfds_py311",
    "top_level_entries": [
      "bair_robot_pushing_small",
      "downloads"
    ],
    "torch_available": false,
    "training_env": "env_isaaclab"
  },
  "data_dir": "/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_tfds",
  "dataset_info": {
    "dataset_size": "20.80 GiB",
    "download_size": "30.06 GiB",
    "features": "Sequence({\n    'action': Tensor(shape=(4,), dtype=float32),\n    'endeffector_pos': Tensor(shape=(3,), dtype=float32),\n    'image_aux1': Image(shape=(64, 64, 3), dtype=uint8),\n    'image_main': Image(shape=(64, 64, 3), dtype=uint8),\n})",
    "name": "bair_robot_pushing_small",
    "splits": {
      "test": {
        "num_examples": 256,
        "num_shards": 1
      },
      "train": {
        "num_examples": 43264,
        "num_shards": 256
      }
    },
    "version": "2.0.0"
  },
  "disk_free_after_gib": 189.03736877441406,
  "disk_free_before_gib": 189.03737258911133,
  "download_success": true,
  "elapsed_time_sec": 0.995246410369873
}
```

BAIR_DOWNLOAD_PASS = true
