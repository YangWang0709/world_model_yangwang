# STEP9A Real Video Minimal Dataset Report

## 1. Goal

Step 9A moves the project from `structured_toy` toward a minimal `real_video_minimal` dataset interface:

`real video files / frame folders -> metadata index -> RealVideoClipDataset -> past/future split -> resize / normalization -> DummyVideoEncoder -> token shards -> smoke / pytest / report`.

This stage does not download large models, does not attach V-JEPA or VideoMAE, and does not run a real visual encoder. It validates real-video data plumbing with a dummy encoder only.

## 2. Resource Guard

The resource guard keeps conservative defaults for an RTX 5080 plus 32GB-class RAM machine:

- `max_samples <= 100`
- `clip_len <= 8`
- `image_size <= 224`
- `batch_size <= 2`
- `num_workers = 0` by default

See `docs/REAL_VIDEO_RESOURCE_REPORT.md` for the current host summary.

## 3. Files Added or Updated

- `configs/real_video_minimal.yaml`
- `configs/token_extraction_real_video_dummy.yaml`
- `data/real_video_dataset.py`
- `data/real_video_index.py`
- `data/real_video_transforms.py`
- `scripts/build_real_video_index.py`
- `scripts/create_real_video_minimal_subset.py`
- `scripts/check_resource_limits.py`
- `scripts/smoke_test_real_video_dataset.py`
- `scripts/smoke_test_real_video_token_extraction.py`
- `scripts/extract_tokens.py`
- `docs/REAL_VIDEO_DATASET_SCHEMA.md`
- `docs/REAL_VIDEO_SMOKE_REPORT.md`
- `docs/REAL_VIDEO_RESOURCE_REPORT.md`
- `tests/test_real_video_index.py`
- `tests/test_real_video_dataset.py`
- `tests/test_real_video_token_extraction.py`
- `tests/test_resource_limits.py`
- `.gitignore`

## 4. Real Video Metadata Schema

`metadata.jsonl` stores one JSON object per sample with:

- `sample_id`
- `source_type`
- `path`
- `task_text`
- `num_frames`
- `fps`
- `height`
- `width`
- `source`
- `split`

See `docs/REAL_VIDEO_DATASET_SCHEMA.md`.

## 5. RealVideoClipDataset

`RealVideoClipDataset` supports:

- `.pt` clip payloads
- frame folders containing `.png`, `.jpg`, or `.jpeg`
- optional video files when a lightweight decode backend is available

The dataset returns `past_video` and `future_video` tensors with shape `[4, 3, 224, 224]` by default.

## 6. Dummy Token Extraction

`scripts/extract_tokens.py` now supports `dataset.name: real_video_minimal`. It reuses the Step 3 token shard schema and the existing `DummyVideoEncoder`, producing local-only shards under `data/token_shards/real_video_dummy`.

This is a data-pipeline smoke path, not a real visual-token encoder.

## 7. Smoke Test Result

See `docs/REAL_VIDEO_SMOKE_REPORT.md`.

Final Step 9A smoke result:

- `REAL_VIDEO_DATASET_SMOKE_PASS = true`
- `REAL_VIDEO_TOKEN_EXTRACTION_SMOKE_PASS = true`
- `real_video_minimal` samples: `16`
- first sample past/future shape: `[4, 3, 224, 224]`
- dummy token shards: `2`
- first shard past/future token shape: `[8, 196, 768]`

## 8. Pytest Result

Final validation:

- `python -m pytest tests -q`
- result: `38 passed`

## 9. Git Commit

- Branch: `feature/tgpawb-step3-token-extraction`
- Commit hash: recorded in the final response and local Step 9A summary after commit creation.
- Push status: recorded in the final response and local Step 9A summary.

No PR is created for this stage.

## 10. What Was Not Done

- no large real dataset download
- no real V-JEPA / VideoMAE encoder
- no large model download
- no VLM grounding
- no RL / policy optimization
- no generated video files committed
- no generated `.pt` files committed
- no checkpoint committed
- no password or token saved
- no PR created

## 11. Next Step Recommendation

Step 9B can connect a frozen VideoMAE or V-JEPA wrapper to either a user-provided small video directory or a chosen tiny public subset, still capped at `<=100` clips until the real encoder path is proven stable.
