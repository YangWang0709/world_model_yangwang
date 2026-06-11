# STEP11A BAIR Robot Pushing Small Dataset Report

## 1. Goal

Step 11A is the first public real-dataset integration stage for the Task-Grounded Predictive Attention Bottleneck world model pipeline. It adds BAIR Robot Pushing small capability checks, an isolated TFDS download environment, a bounded subset exporter, a PyTorch loader, smoke coverage, and pytest coverage.

Status: completed after Step 11A-fix.

## 2. Cloud Server Decision

No cloud server was required. The local Ubuntu RTX 5080 host completed the work. This stage mainly used network, disk, and CPU; GPU was not a bottleneck.

- disk threshold: at least 80 GiB free
- final free disk: about 189.04 GiB
- final available RAM: about 25.2 GiB
- cloud recommendation: not needed

## 3. Dependency Isolation

The original blocker was that `env_isaaclab` did not have `tensorflow` or `tensorflow_datasets`. To avoid polluting the training environment, Step 11A-fix created a separate conda environment:

- TFDS env: `tgpawb_tfds_py311`
- Python: `3.11.15`
- TensorFlow: `2.21.0`
- TensorFlow Datasets: `4.9.10`
- TF GPU devices: `[]`
- training env: `env_isaaclab`
- `env_isaaclab` pollution: `false`

`env_isaaclab` still has no TensorFlow or TensorFlow Datasets installed and is used only for loader smoke and pytest.

## 4. Dataset Facts

- Dataset: BAIR Robot Pushing small
- Version: `2.0.0`
- Resolution: 64x64 source frames
- Download size: `30.06 GiB`
- Prepared dataset size: `20.80 GiB`
- Splits: train/test
- Train examples: `43264`
- Test examples: `256`
- Fields: `image_main`, `image_aux1`, `action`, `endeffector_pos`

## 5. Files Added or Updated

- `.gitignore`
- `configs/bair_robot_pushing_small.yaml`
- `configs/bair_robot_pushing_subset.yaml`
- `data/__init__.py`
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

## 6. Download

- data_dir: `/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_tfds`
- download_success: `true`
- blocking_reason: `None`
- can_attempt_download: `true`
- final disk_free_before_gib: about `189.04`
- final disk_free_after_gib: about `189.04`
- refreshed elapsed_time_sec: about `0.995`

The raw tar download initially hit a network `IncompleteRead` at about 38 percent. The fix resumed the already downloaded 12.2 GB partial file with HTTP Range support using `curl -C -`, completed the tar, verified SHA256, wrote the TFDS cache `.INFO`, and then let TFDS prepare the dataset from local cache without starting from zero.

The verified tar:

- size: `32274964480` bytes
- SHA256: `c6f8d164ab2be07174ef597e3e070d7466f3142caddc467b6af119b9458cd9cd`

## 7. Subset Export

- output directory: `/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset`
- train exported: `100`
- test exported: `16`
- skipped_short: `0`
- clip format: `.pt` payload with `[T, C, H, W]` float video and optional action/state tensors
- metadata format: per-split `metadata.jsonl`
- camera: `image_main`
- action included: `true`
- endeffector_pos included: `true`
- export_success: `true`
- elapsed_time_sec: about `2.48`

## 8. Loader

`BAIRRobotPushingDataset` reads exported `.pt` clips and does not import TensorFlow. It returns:

- `past_video`: `[4, 3, 224, 224]`
- `future_video`: `[4, 3, 224, 224]`
- `actions`: `[8, 4]`
- `endeffector_pos`: `[8, 3]`
- `task_text`, `sample_id`, and `metadata`

## 9. Smoke Test Result

- `BAIR_DATASET_SMOKE_PASS = true`
- download_status: `skipped_subset_ready`
- subset_export_status: `reused_existing`
- train subset size: `100`
- test subset size: `16`
- past shape: `[4, 3, 224, 224]`
- future shape: `[4, 3, 224, 224]`
- action shape: `[8, 4]`
- value range: roughly `[0.0213, 1.0]`

See `docs/BAIR_DATASET_SMOKE_REPORT.md`.

## 10. Pytest Result

Remote pytest passed in `env_isaaclab`:

```text
62 passed in 1.46s
```

The BAIR tests do not download BAIR data and do not require TensorFlow.

## 11. Git Commit

Branch: `feature/tgpawb-step3-token-extraction`. Commit hash and push status are recorded in the final response and local Step 11A summary after commit/push. No PR is created.

## 12. What Was Not Done

- no new model download
- no V-JEPA download
- no VideoMAE training
- no Teacher training
- no Student training
- no token extraction yet
- no predictive importance generation
- no VLM grounding
- no RL / policy optimization
- no dataset files committed
- no generated `.pt` clips committed
- no checkpoint committed
- no password or token saved
- no PR created

## 13. Next Step Recommendation

Step 11B should use the existing local VideoMAE checkpoint to extract frozen encoder tokens from the BAIR exported subset, starting with at most 100 train and 16 test clips, `batch_size=1`, and `num_workers=0`.
