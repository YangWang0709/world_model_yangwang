# BAIR Dataset Smoke Report

Command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_bair_dataset.py`

- BAIR_DATASET_SMOKE_PASS: `false`
- blocking_reason: `tensorflow_datasets is not available`
- download_status: `blocked`
- subset_export_status: `blocked`
- train subset size: `0`
- test subset size: `0`
- past shape: `None`
- future shape: `None`
- actions shape: `None`
- value range: `None`
- disk_free_before_gib: `272.66502380371094`
- disk_free_after_gib: `272.66502380371094`
- elapsed_time_sec: `0.00882411003112793`

```json
{
  "actions_shape": null,
  "blocking_reason": "tensorflow_datasets is not available",
  "capability_summary": {
    "bair_data_dir_exists": false,
    "blocking_reason": "tensorflow_datasets is not available",
    "can_attempt_download": false,
    "data_dir": "/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_tfds",
    "disk_free_gib": 272.66502380371094,
    "download_allowed": true,
    "min_disk_required_gib": 80.0,
    "numpy_available": true,
    "partial_tfds_download_detected": false,
    "proxy_env": {},
    "python_executable": "/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python",
    "python_version": "3.11.15",
    "ram_available_gib": 24.663436889648438,
    "tensorflow_available": false,
    "tensorflow_datasets_available": false,
    "tensorflow_datasets_version": null,
    "tensorflow_version": null,
    "tfds_builder_available": false,
    "tfds_builder_error": null,
    "top_level_entries": [],
    "torch_available": true
  },
  "data_dir": "/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_tfds",
  "disk_free_after_gib": 272.66502380371094,
  "disk_free_before_gib": 272.66502380371094,
  "download_status": "blocked",
  "download_summary": {
    "attempts": 0,
    "blocking_reason": "tensorflow_datasets is not available",
    "capability_summary": {
      "bair_data_dir_exists": false,
      "blocking_reason": "tensorflow_datasets is not available",
      "can_attempt_download": false,
      "data_dir": "/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_tfds",
      "disk_free_gib": 272.66502380371094,
      "download_allowed": true,
      "min_disk_required_gib": 80.0,
      "numpy_available": true,
      "partial_tfds_download_detected": false,
      "proxy_env": {},
      "python_executable": "/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python",
      "python_version": "3.11.15",
      "ram_available_gib": 24.663436889648438,
      "tensorflow_available": false,
      "tensorflow_datasets_available": false,
      "tensorflow_datasets_version": null,
      "tensorflow_version": null,
      "tfds_builder_available": false,
      "tfds_builder_error": null,
      "top_level_entries": [],
      "torch_available": true
    },
    "data_dir": "/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_tfds",
    "dataset_info": null,
    "disk_free_after_gib": 272.66502380371094,
    "disk_free_before_gib": 272.66502380371094,
    "download_success": false,
    "elapsed_time_sec": 0.0012025833129882812
  },
  "elapsed_time_sec": 0.00882411003112793,
  "endeffector_pos_shape": null,
  "future_shape": null,
  "past_shape": null,
  "smoke_pass": false,
  "subset_export_status": "blocked",
  "subset_export_summary": {
    "blocking_reason": "tensorflow_datasets is not available",
    "capability_summary": {
      "tensorflow_available": false,
      "tensorflow_datasets_available": false,
      "tensorflow_datasets_error": "tensorflow_datasets is not installed",
      "tensorflow_datasets_version": null,
      "tensorflow_error": "tensorflow is not installed",
      "tensorflow_version": null,
      "tfds_builder_available": false,
      "tfds_builder_error": null
    },
    "data_dir": "/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_tfds",
    "elapsed_time_sec": 6.771087646484375e-05,
    "export_success": false,
    "output_root": "/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset",
    "splits": {},
    "tfds_name": "bair_robot_pushing_small",
    "tfds_version": "2.0.0"
  },
  "subset_output_root": "/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset",
  "test_subset_size": 0,
  "train_subset_size": 0,
  "value_range": null
}
```

BAIR_DATASET_SMOKE_PASS = false
