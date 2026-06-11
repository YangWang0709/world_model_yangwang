# STEP11D Predictive Importance on BAIR VideoMAE Tokens

## 1. Goal

Step 11D generates Teacher predictive token importance labels on the public BAIR Robot Pushing small subset using the BAIR VideoMAE token shards from Step 11B and the full-token BAIR Teacher checkpoint from Step 11C.

The labels are prepared for the next Student selector smoke stage. This remains a bounded smoke on a small public robot video subset, not a paper-scale result.

## 2. Cloud Server Decision

This stage does not require a cloud server. It does not run the VideoMAE encoder, download models, download datasets, or train any model. It only reads existing token shards and a small Teacher checkpoint, then performs chunked Teacher occlusion on the local RTX 5080.

If CUDA OOM or a resource warning appears, the recommended next action is to reduce `token_chunk_size` or move larger future runs to a 4090 / 48GB class machine.

## 3. Input

- train token shard dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train`
- test token shard dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test`
- teacher checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1/checkpoints/teacher_world_model_step_000300.pt`
- train samples: `100`
- test samples: `16`
- token shape: `[B, 392, 768]`
- source encoder: `VideoMAE`

## 4. Formula

Predictive importance is computed by Teacher token occlusion:

```text
I_i = Loss(Teacher(z_without_i), z_future) - Loss(Teacher(z_full), z_future)
```

`z_future` is mean-pooled from future tokens to match the full-token Teacher target.

## 5. Generation Setup

- config: `configs/generate_importance_bair_videomae_smoke.yaml`
- method: `teacher_token_occlusion`
- batch size: `1`
- token chunk size: `32`
- mask mode: `zero`
- normalization: `minmax_per_sample`
- output root: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke`
- train output dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/train`
- test output dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/test`

## 6. Smoke Test Result

The live smoke result is recorded in `docs/IMPORTANCE_BAIR_VIDEOMAE_SMOKE_REPORT.md`.

Result:

```text
IMPORTANCE_BAIR_VIDEOMAE_SMOKE_PASS = true
```

- train importance shards: `100`
- test importance shards: `16`
- total samples: `116`
- num tokens: `392`
- token dim: `768`
- token chunk size: `32`
- overall importance mean/std/min/max: `1.2047740710841026e-05 / 0.0010329956421628594 / -0.008821845054626465 / 0.007312774658203125`
- overall normalized importance mean/std/min/max: `0.5080463886260986 / 0.19772428274154663 / 0.0 / 1.0`
- overall base loss mean: `0.8841761946678162`
- overall masked loss mean: `0.884188175201416`
- generation elapsed time sec: `0.639`
- smoke elapsed time sec: `0.705`
- OOM: `false`
- cloud recommendation: not required for Step 11D

## 7. Pytest Result

The full test suite passed:

```text
70 passed in 1.39s
```

Added:

- `tests/test_importance_bair_videomae_config.py`
- `tests/test_importance_bair_videomae_shapes.py`

The tests do not download models, download BAIR, read real BAIR shards, or write project artifacts.

## 8. Git Commit

- repo: `git@github.com:YangWang0709/world_model_yangwang.git`
- branch: `feature/tgpawb-step3-token-extraction`
- commit: recorded after commit/push in the final run summary
- PR: not created

## 9. What Was Not Done

- no new model download
- no new dataset download
- no BAIR re-download
- no VideoMAE extraction rerun
- no VideoMAE training
- no Teacher retraining
- no Student training
- no TokenCompressor / StudentWorldModel wiring
- no VLM grounding
- no RL / policy optimization
- no model weight committed
- no BAIR data committed
- no token shard committed
- no importance shard committed
- no checkpoint committed
- no password or token saved
- no PR created

## 10. Next Step Recommendation

Step 11E should train a BAIR Student selector smoke using the predictive importance labels generated here. The result should still be described as a small public robot-video smoke until the data scale and evaluation are widened.
