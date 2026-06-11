# STEP11B BAIR VideoMAE Token Extraction Report

## 1. Goal

Step 11B extracts frozen VideoMAE tokens for the Step 11A exported BAIR Robot Pushing small subset. This connects the first real public robotics video benchmark to the existing offline token-shard pipeline.

## 2. Cloud Server Decision

No cloud server was needed for this stage. The smoke ran on the local Ubuntu RTX 5080 host with conservative settings:

- batch_size: 1
- num_workers: 0
- train samples: 100
- test samples: 16
- shard_size: 4

The smoke completed without OOM. A cloud 4090 / 48GB server is only recommended for future larger BAIR runs, multiple seeds, more baselines, or if batch_size=1 starts to OOM.

## 3. Input Dataset

- BAIR subset path: `/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset`
- train metadata: `/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset/train/metadata.jsonl`
- test metadata: `/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset/test/metadata.jsonl`
- train samples: 100
- test samples: 16
- input clip split: past `[4, 3, 224, 224]`, future `[4, 3, 224, 224]`
- action metadata: included, shape `[8, 4]`
- endeffector metadata: included, shape `[8, 3]`

## 4. Encoder

- encoder: VideoMAE
- model path: `/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local`
- cache dir: `/home/ubuntu22/tgpawb_world_model/model_cache/huggingface`
- allow_download: false
- local_files_only: true
- fallback: disabled
- used_fallback: false
- mode: frozen eval with `torch.no_grad()`
- output mode: `last_hidden_state`
- effective frames: 4
- token shape per sample: `[392, 768]`

## 5. Extraction Setup

- config: `configs/token_extraction_bair_videomae_smoke.yaml`
- script: `scripts/extract_tokens.py`
- smoke: `scripts/smoke_test_bair_videomae_token_extraction.py`
- output root: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke`
- train output: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train`
- test output: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test`
- train shards: 25
- test shards: 4
- total shards: 29

Token shards are generated under ignored `data/token_shards/` and are not committed.

## 6. Smoke Test Result

The smoke report is `docs/BAIR_VIDEOMAE_TOKEN_EXTRACTION_SMOKE_REPORT.md`.

- `BAIR_VIDEOMAE_TOKEN_EXTRACTION_SMOKE_PASS = true`
- first train shard shape: `[4, 392, 768]`
- first test shard shape: `[4, 392, 768]`
- elapsed_time_sec: 4.323
- OOM: false
- GPU memory before: 815 MiB used
- GPU memory after: 1542 MiB used
- RAM before: 5.610 GiB used
- RAM after: 6.269 GiB used

## 7. Pytest Result

Full test suite result after Step 11B:

```text
64 passed in 1.40s
```

The new pytest coverage validates the bounded BAIR VideoMAE config and a BAIR-like multi-split extraction path with a fake encoder. Pytest does not run real VideoMAE, does not read real BAIR, and does not download models.

## 8. Git Commit

- branch: `feature/tgpawb-step3-token-extraction`
- commit hash: recorded by `git rev-parse HEAD` after the final Step 11B commit
- push status: pushed after validation
- PR: not created

## 9. What Was Not Done

- no new model download
- no new dataset download
- no BAIR re-download
- no VideoMAE training
- no Teacher training
- no Student training
- no predictive importance generation
- no VLM grounding
- no RL / policy optimization
- no `/home/ubuntu22/VLA` changes
- no `/home/ubuntu22/MapExRL` changes
- no `/home/ubuntu22/test_gpt` changes
- no model weight committed
- no BAIR data committed
- no `.pt` clips committed
- no token shard committed
- no checkpoint committed
- no password or token saved
- no PR created
- no merge to main

## 10. Next Step Recommendation

Step 11C should train a tiny Teacher smoke on the BAIR VideoMAE token shards. Keep the same bounded data scale first: train split 100 samples, test split 16 samples, conservative batch size, and no expansion until the tiny public-data teacher path is verified.
