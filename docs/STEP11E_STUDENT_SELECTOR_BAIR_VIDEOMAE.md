# STEP11E Student Selector on BAIR VideoMAE Importance

## 1. Goal

Step 11E trains a Student `AttentionSelector` smoke on BAIR VideoMAE past tokens using the predictive importance labels generated in Step 11D.

The selector maps BAIR past tokens with shape `[B, 392, 768]` to per-token score logits `[B, 392]`. Sigmoid scores are regressed to Teacher normalized importance labels `[B, 392]`.

This is a bounded smoke stage, not a final paper-scale result.

## 2. Cloud Server Decision

This stage does not require a cloud server. It does not run the VideoMAE encoder, download any model, download any dataset, retrain the Teacher, train a StudentWorldModel, or run policy/RL code.

If CUDA OOM or a resource warning appears when scaling beyond this smoke, the next larger run should use a 4090 / 48GB class machine or reduce batch size.

## 3. Input

- train token dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train`
- test token dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test`
- train importance dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/train`
- test importance dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/test`
- train samples: `100`
- test samples: `16`
- token shape: `[B, 392, 768]`
- importance shape: `[B, 392]`
- source encoder: `VideoMAE`

## 4. Training Setup

- model: `AttentionSelector`
- target: `importance_scores_norm`
- loss: MSE
- key token mask: not required
- topK: `16`
- evaluation: Teacher-importance regression, Pearson correlation, and Teacher topK overlap

## 5. Smoke Test Result

The live result is recorded in `docs/STUDENT_SELECTOR_BAIR_VIDEOMAE_SMOKE_REPORT.md`.

Result:

```text
STUDENT_SELECTOR_BAIR_VIDEOMAE_SMOKE_PASS = true
```

- run dir: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1`
- checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/checkpoints/student_selector_step_000100.pt`
- num steps: `100`
- initial loss: `0.043159741908311844`
- final loss: `0.029176875948905945`
- best loss: `0.02254447154700756`
- loss decreased: `true`
- train importance MSE: `0.03457484021782875`
- train Pearson correlation: `0.3355245590209961`
- train target topK overlap: `0.16875`
- test importance MSE: `0.03200181573629379`
- test importance MAE: `0.14211484789848328`
- test Pearson correlation: `0.2978422939777374`
- test target top1 overlap: `0.0`
- test target topK overlap: `0.1640625`
- test selected teacher importance mean: `0.5927466750144958`
- test random teacher importance mean: `0.4584925174713135`
- test selected-vs-random importance gap: `0.13425415754318237`
- OOM: `false`
- cloud recommendation: not required for Step 11E

## 6. Pytest Result

The full test suite passed:

```text
75 passed in 1.40s
```

## 7. Git Commit

- repo: `git@github.com:YangWang0709/world_model_yangwang.git`
- branch: `feature/tgpawb-step3-token-extraction`
- commit: recorded after commit/push in the final run summary
- PR: not created

## 8. What Was Not Done

- no new model download
- no new dataset download
- no BAIR re-download
- no VideoMAE extraction rerun
- no VideoMAE training
- no Teacher retraining
- no StudentWorldModel training
- no TokenCompressor
- no VLM grounding
- no RL / policy optimization
- no model weight committed
- no BAIR data committed
- no token shard committed
- no importance shard committed
- no checkpoint committed
- no password or token saved
- no PR created

## 9. Next Step Recommendation

Step 11F should connect the BAIR selector with `TokenCompressor` and `StudentWorldModel`, then train a compressed Student future-prediction smoke on BAIR VideoMAE tokens.
