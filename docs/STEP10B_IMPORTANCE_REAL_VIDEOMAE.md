# STEP10B Predictive Importance on Real VideoMAE Tokens

## 1. Goal

Step 10B generates teacher-occlusion predictive token importance labels on the real VideoMAE token shards produced in Step 9C, using the Step 10A full-token TeacherWorldModel checkpoint.

This is a bounded smoke validation on 8 samples, not a paper-scale result.

## 2. Cloud Server Decision

No cloud server was required for this stage. Step 10B did not run the VideoMAE encoder and did not train a model. It only read 8 existing token samples and a tiny Teacher checkpoint on the local RTX 5080 server.

The smoke completed without OOM. GPU memory moved from 0.954 GiB to 1.028 GiB used, and RAM moved from 4.224 GiB to 4.785 GiB used. A larger 4090 / 48GB server is not required for this bounded 8-sample smoke, but it remains the right direction if later steps scale the sample count or token chunk size.

## 3. Input

- token shard dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke`
- Teacher checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/teacher_real_video_videomae_smoke_v1/checkpoints/teacher_world_model_step_000100.pt`
- samples: `8`
- source shards: `4`
- source token shape per shard: `[2, 784, 768]`
- source encoder: `videomae`
- used fallback: `false`

## 4. Formula

Predictive importance is computed as:

```text
I_i = Loss(Teacher(z_without_i), z_future) - Loss(Teacher(z_full), z_future)
```

For each source sample, the full-token base loss is computed first. Then each past token is zero-masked in chunks, the masked loss is recomputed, and the loss delta becomes the token importance score.

## 5. Generation Setup

- config: `configs/generate_importance_real_video_videomae_smoke.yaml`
- method: `teacher_token_occlusion`
- batch_size: `1`
- token_chunk_size: `16`
- max_samples: `8`
- mask mode: `zero`
- normalization: `minmax_per_sample`
- output dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke`

The generator slices each source shard into batch-size-limited chunks and never constructs a full `[B, N, N, D]` tensor. For Step 10B it builds only `[B * C, N, D]` masked batches where `C=16`.

## 6. Smoke Test Result

Command:

```bash
/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_predictive_importance_real_video_videomae.py
```

Result:

- `IMPORTANCE_REAL_VIDEOMAE_SMOKE_PASS = true`
- generated importance shards: `8`
- first shard importance shape: `[1, 784]`
- first shard normalized importance shape: `[1, 784]`
- first shard base losses shape: `[1]`
- first shard masked losses shape: `[1, 784]`
- OOM: `false`
- elapsed time: `0.185` sec in the smoke resource summary

Aggregate eval:

- importance mean: `1.563841465213045e-06`
- importance std: `0.00022119932691566646`
- importance min: `-0.003388732671737671`
- importance max: `0.0024299323558807373`
- normalized importance mean: `0.46076127886772156`
- normalized importance std: `0.33224743604660034`
- normalized importance min: `0.0`
- normalized importance max: `1.0`
- positive importance ratio: `0.4832589328289032`
- base loss mean: `0.1661720871925354`
- masked loss mean: `0.16617366671562195`
- top1 importance mean: `0.0008337665349245071`
- top5 importance mean: `0.0006721671088598669`
- top10 importance mean: `0.000566251517739147`

The full smoke report is in `docs/IMPORTANCE_REAL_VIDEOMAE_SMOKE_REPORT.md`.

## 7. Pytest Result

Full pytest was run after the Step 10B smoke:

```text
49 passed in 1.23s
```

The new tests cover the real VideoMAE importance config bounds and `[1, 784, 768]` shape handling with `token_chunk_size=16`.

## 8. Git Commit

- branch: `feature/tgpawb-step3-token-extraction`
- repository: `git@github.com:YangWang0709/world_model_yangwang.git`
- PR: not created
- main merge: not performed

The exact commit hash is recorded after commit/push in the local Step 10B summary and final handoff.

## 9. What Was Not Done

- no new model download
- no new real dataset download
- no VideoMAE encoder rerun
- no VideoMAE training
- no Teacher retraining
- no Student training
- no VLM grounding
- no RL / policy optimization
- no large dependency installation
- no model weight committed
- no generated token shard `.pt` committed
- no generated importance shard `.pt` committed
- no checkpoint committed
- no password or token saved
- no PR created

## 10. Next Step Recommendation

Step 10C can train a small Student selector smoke on the real VideoMAE importance labels produced here. Because the current dataset has only 8 samples, Step 10C should remain a smoke test and should not be described as a final paper result.

If later work scales from this smoke to a real dataset subset, the next data step should explicitly download or stage a small approved BAIR / ManiSkill subset and revisit resource planning before larger extraction or importance generation.
