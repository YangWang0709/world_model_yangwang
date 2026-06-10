# STEP3 Token Extraction Pipeline Report

## 1. Goal

Step 3 implements the data-reading interface, token shard schema, and offline dummy token extraction pipeline.

This step does not train models, download V-JEPA / VideoMAE / VLM weights, download datasets, or install large dependencies.

## 2. Files Added or Updated

Added or updated files:

- `.gitignore`
- `configs/token_extraction_dummy.yaml`
- `configs/token_shard_schema.yaml`
- `data/__init__.py`
- `data/datasets.py`
- `data/toy_data.py`
- `data/token_shards.py`
- `data/video_clip_dataset.py`
- `scripts/create_toy_video_dataset.py`
- `scripts/extract_tokens.py`
- `scripts/inspect_token_shard.py`
- `scripts/smoke_test_token_extraction.py`
- `docs/TOKEN_SHARD_SCHEMA.md`
- `docs/TOKEN_EXTRACTION_SMOKE_REPORT.md`
- `docs/STEP3_TOKEN_EXTRACTION_PIPELINE.md`
- `tests/test_video_clip_dataset.py`
- `tests/test_token_shards.py`
- `tests/test_token_extraction_pipeline.py`

Step 2 files and tests were preserved.

## 3. Toy Dataset

Toy dataset path:

`/home/ubuntu22/tgpawb_world_model/data/toy_videos`

The toy dataset generator creates 16 samples by default. Each clip is a `.pt` dict containing:

- `video`: torch tensor `[T_total, C, H, W]`, default `[6, 3, 64, 64]`, float32 in `[0, 1]`
- `task_text`
- `sample_id`
- `fps`
- `source`

The metadata manifest is `metadata.jsonl`, with one JSON object per sample. The generated toy dataset is a local smoke-test artifact and is ignored by git.

## 4. VideoClipDataset

`VideoClipDataset` reads `metadata.jsonl`, loads `.pt` clip payloads, validates the video tensor, and returns a past/future split.

Default split:

- `past_len = 4`
- `future_len = 2`

Returned fields:

- `past_video`: `[past_len, C, H, W]`
- `future_video`: `[future_len, C, H, W]`
- `task_text`
- `sample_id`
- `metadata`

Invalid paths, malformed video tensors, and clips with insufficient frames raise explicit errors.

## 5. Token Shard Schema

Schema version: `0.1.0`

Required keys:

- `schema_version`
- `encoder_name`
- `encoder_config`
- `created_at`
- `split`
- `sample_ids`
- `task_texts`
- `past_tokens`
- `future_tokens`
- `metadata`

Tensor shapes:

- `past_tokens`: `[B, N, D]`
- `future_tokens`: `[B, N, D]` or `[B, D]`

For the dummy encoder smoke test, the first shard has:

- `past_tokens`: `[8, 196, 768]`
- `future_tokens`: `[8, 196, 768]`

Token shards are generated artifacts under `data/token_shards/` and are ignored by git.

## 6. Extraction Pipeline

`scripts/extract_tokens.py` implements:

`VideoClipDataset -> DummyVideoEncoder -> past_tokens/future_tokens -> token shard writer`

The default config is `configs/token_extraction_dummy.yaml`. It uses the dummy encoder only. If a non-dummy encoder is requested, the script raises `NotImplementedError` and explains that Step 3 does not download large model weights.

## 7. Smoke Test Result

Smoke report: `docs/TOKEN_EXTRACTION_SMOKE_REPORT.md`

- dataset size: `16`
- generated shard count: `2`
- generated shard files: `tokens_shard_000000.pt`, `tokens_shard_000001.pt`
- first shard `schema_version`: `0.1.0`
- first shard `encoder_name`: `dummy_video_encoder`
- first shard `num_samples`: `8`
- first shard `past_tokens_shape`: `[8, 196, 768]`
- first shard `future_tokens_shape`: `[8, 196, 768]`
- split: `toy`
- `TOKEN_EXTRACTION_SMOKE_PASS = true`

## 8. Pytest Result

All tests passed in the remote `env_isaaclab` conda environment:

`6 passed`

This includes the original Step 2 tests and the new Step 3 dataset, shard, and extraction pipeline tests.

## 9. Git Commit

- Repository: `YangWang0709/VLA-yangwang`
- Branch: `feature/tgpawb-step3-token-extraction`
- Commit message: `feat(tgpawb): add token extraction pipeline`
- Commit hash: recorded after commit in `STEP3_LOCAL_SUMMARY.md` and the final user response.
- Remote push status: recorded after push attempt in `STEP3_LOCAL_SUMMARY.md` and the final user response.

## 10. What Was Not Done

- No training.
- No V-JEPA / VideoMAE download.
- No VLM download.
- No dataset download.
- No large dependency install.
- No modification to `/home/ubuntu22/VLA`, `/home/ubuntu22/MapExRL`, or `/home/ubuntu22/test_gpt`.
- No password or token saved.
- Generated toy data and token shards were not committed.

## 11. Next Step Recommendation

Recommended Step 4: implement the real full-token `TeacherWorldModel` training script, but first use dummy token shards for a 10-step tiny overfit or sanity run. After that, connect a real frozen encoder path while preserving the Step 3 token shard reader contract.

