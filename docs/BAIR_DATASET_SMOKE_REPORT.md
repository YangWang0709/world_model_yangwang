# BAIR Dataset Smoke Report

Command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_bair_dataset.py`

- BAIR_DATASET_SMOKE_PASS: `true`
- blocking_reason: `None`
- download_status: `skipped_subset_ready`
- subset_export_status: `reused_existing`
- train subset size: `100`
- test subset size: `16`
- past shape: `[4, 3, 224, 224]`
- future shape: `[4, 3, 224, 224]`
- actions shape: `[8, 4]`
- value range: `{'past_min': 0.021308593451976776, 'past_max': 1.0, 'future_min': 0.02543008141219616, 'future_max': 1.0}`
- disk_free_before_gib: `189.03736877441406`
- disk_free_after_gib: `189.03736877441406`
- elapsed_time_sec: `0.014220237731933594`

```json
{
  "actions_shape": [
    8,
    4
  ],
  "blocking_reason": null,
  "capability_summary": {
    "bair_data_dir_exists": true,
    "blocking_reason": "tensorflow_datasets is not available",
    "can_attempt_download": false,
    "current_conda_env": "env_isaaclab",
    "data_dir": "/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_tfds",
    "disk_free_gib": 189.03736877441406,
    "download_allowed": true,
    "env_isolation": {
      "tensorflow_expected_in_training_env": false,
      "training_env_not_modified": true,
      "uses_isolated_tfds_env": false
    },
    "min_disk_required_gib": 80.0,
    "numpy_available": true,
    "partial_tfds_download_detected": false,
    "proxy_env": {},
    "python_executable": "/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python",
    "python_version": "3.11.15",
    "ram_available_gib": 25.232540130615234,
    "tensorflow_available": false,
    "tensorflow_datasets_available": false,
    "tensorflow_datasets_version": null,
    "tensorflow_version": null,
    "tfds_builder_available": false,
    "tfds_builder_error": null,
    "tfds_env": "tgpawb_tfds_py311",
    "top_level_entries": [
      "bair_robot_pushing_small",
      "downloads"
    ],
    "torch_available": true,
    "training_env": "env_isaaclab"
  },
  "data_dir": "/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_tfds",
  "disk_free_after_gib": 189.03736877441406,
  "disk_free_before_gib": 189.03736877441406,
  "download_status": "skipped_subset_ready",
  "download_summary": null,
  "elapsed_time_sec": 0.014220237731933594,
  "endeffector_pos_shape": [
    8,
    3
  ],
  "future_shape": [
    4,
    3,
    224,
    224
  ],
  "past_shape": [
    4,
    3,
    224,
    224
  ],
  "sample_id": "bair_train_000000",
  "smoke_pass": true,
  "subset_export_status": "reused_existing",
  "subset_export_summary": null,
  "subset_output_root": "/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset",
  "test_subset_size": 16,
  "train_subset_size": 100,
  "value_range": {
    "future_max": 1.0,
    "future_min": 0.02543008141219616,
    "past_max": 1.0,
    "past_min": 0.021308593451976776
  }
}
```

BAIR_DATASET_SMOKE_PASS = true
