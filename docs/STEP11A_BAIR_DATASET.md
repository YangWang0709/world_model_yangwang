# STEP11A BAIR Robot Pushing Small Dataset Report

## 1. Goal

Step 11A is the first public real-dataset integration stage for the Task-Grounded Predictive Attention Bottleneck world model pipeline. It adds BAIR Robot Pushing small download capability checks, a bounded subset exporter, a PyTorch loader, smoke coverage, and pytest coverage.

Status: partially completed. The code, configs, reports, and tests are in place, but the remote environment cannot attempt the actual BAIR download yet because `tensorflow_datasets` and `tensorflow` are not installed.

## 2. Cloud Server Decision

This stage does not require a cloud server by default. It is dominated by network and disk use rather than GPU compute. The local RTX 5080 server had enough disk for this stage.

- disk threshold: at least 80 GiB free
- observed free disk before Step 11A download attempt: about 272.66 GiB
- observed available RAM: about 24.66 GiB
- cloud recommendation: not needed for the current blocker

The next action is to prepare the local environment with the required TFDS/TensorFlow dataset tooling or choose an approved lighter BAIR reader path. Cloud is only useful if local network/disk remains blocked after dependencies are resolved.

## 3. Dataset Facts

- Dataset: BAIR Robot Pushing small
- Version: 2.0.0
- Resolution: 64x64 source frames
- Approximate download size: 30 GiB
- Approximate prepared dataset size: 20 GiB
- Splits: train/test
- Expected fields: `image_main`, optional `image_aux1`, `action`, and `endeffector_pos` or state-like observations

## 4. Files Added or Updated

- `.gitignore`
- `configs/bair_robot_pushing_small.yaml`
- `configs/bair_robot_pushing_subset.yaml`
- `data/bair_tfds_utils.py`
- `data/bair_subset_export.py`
- `data/bair_dataset.py`
- `scripts/check_bair_dataset_capabilities.py`
- `scripts/download_bair_robot_pushing_small.py`
- `scripts/export_bair_subset.py`
- `scripts/smoke_test_bair_dataset.py`
- `tests/test_bair_config.py`
- `tests/test_bair_subset_export_schema.py`
- `tests/test_bair_dataset_loader.py`
- `tests/test_bair_capability_report.py`
- `docs/BAIR_DATASET_SCHEMA.md`
- `docs/BAIR_CAPABILITY_REPORT.md`
- `docs/BAIR_DOWNLOAD_REPORT.md`
- `docs/BAIR_SUBSET_EXPORT_REPORT.md`
- `docs/BAIR_DATASET_SMOKE_REPORT.md`
- `docs/bair_capabilities.json`

## 5. Download

- data_dir: `/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_tfds`
- download_success: `false`
- blocking_reason: `tensorflow_datasets is not available`
- disk_free_before_gib: about `272.665`
- disk_free_after_gib: about `272.665`
- elapsed_time_sec: about `0.001`

No BAIR TFDS files were downloaded.

## 6. Subset Export

- output directory: `/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset`
- train max episodes: `100`
- test max episodes: `16`
- clip format: `.pt` payload with `[T, C, H, W]` float video and optional action/state tensors
- metadata format: per-split `metadata.jsonl`
- export_success: `false`
- blocking_reason: `tensorflow_datasets is not available`

No BAIR `.pt` clips were exported.

## 7. Loader

`BAIRRobotPushingDataset` was implemented and tested on synthetic exported `.pt` clips. It returns:

- `past_video`: `[4, 3, 224, 224]` by default
- `future_video`: `[4, 3, 224, 224]` by default
- `actions`: optional tensor
- `endeffector_pos`: optional tensor
- `task_text`, `sample_id`, and `metadata`

The loader does not import TensorFlow and only reads exported `.pt` clips.

## 8. Smoke Test Result

- `BAIR_DATASET_SMOKE_PASS = false`
- blocking_reason: `tensorflow_datasets is not available`
- download_status: `blocked`
- subset_export_status: `blocked`
- train subset size: `0`
- test subset size: `0`
- first sample shape: not available because no subset was exported

See `docs/BAIR_DATASET_SMOKE_REPORT.md`.

## 9. Pytest Result

Remote pytest passed:

```text
62 passed in 1.32s
```

The new BAIR tests do not download BAIR data.

## 10. Git Commit

Branch: `feature/tgpawb-step3-token-extraction`. Commit hash and push status are recorded in the final response and local Step 11A summary after commit/push. No PR is created.

## 11. What Was Not Done

- no model download
- no VideoMAE training
- no Teacher training
- no Student training
- no token extraction yet
- no VLM grounding
- no RL / policy optimization
- no dataset files committed
- no generated `.pt` committed
- no checkpoint committed
- no password or token saved
- no PR created

## 12. Next Step Recommendation

Step 11A should be resumed by installing or enabling the required dataset tooling in `env_isaaclab`, preferably after explicit approval because full TensorFlow can be large. Once capability passes, run the BAIR download script, export the <=100 train / <=16 test subset, and rerun `scripts/smoke_test_bair_dataset.py`.

After the subset exists, Step 11B should use the existing local VideoMAE checkpoint to extract frozen encoder tokens from the BAIR exported subset, starting with `batch_size=1` and `num_workers=0`.
