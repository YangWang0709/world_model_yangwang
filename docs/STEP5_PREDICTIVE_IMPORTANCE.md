# STEP5 Predictive Token Importance Report

## 1. Goal

Step 5 adds a minimal predictive token importance pipeline using the tiny full-token `TeacherWorldModel` checkpoint from Step 4.
This stage only uses dummy token shards and the tiny trained teacher checkpoint.

## 2. Formula

```text
I_i = Loss(Teacher(z_without_i), z_future) - Loss(Teacher(z_full), z_future)
```

Each token is masked with a zero vector, the teacher prediction loss is recomputed, and the loss delta becomes the token importance label.

## 3. Files Added or Updated

```text
.gitignore
configs/generate_importance_dummy.yaml
configs/importance_shard_schema.yaml
data/importance_shards.py
scripts/generate_predictive_importance.py
scripts/inspect_importance_shard.py
scripts/smoke_test_predictive_importance.py
eval/eval_importance_summary.py
tests/test_importance_shards.py
tests/test_predictive_importance.py
tests/test_importance_pipeline.py
docs/IMPORTANCE_SHARD_SCHEMA.md
docs/STEP5_PREDICTIVE_IMPORTANCE.md
docs/PREDICTIVE_IMPORTANCE_SMOKE_REPORT.md
```

## 4. Importance Shard Schema

Schema version: `0.1.0`.

Required tensor shapes:

```text
importance_scores: [B, N]
importance_scores_norm: [B, N]
base_losses: [B]
masked_losses: [B, N]
```

The schema is implemented in `data/importance_shards.py` and documented in `docs/IMPORTANCE_SHARD_SCHEMA.md`.

## 5. Generation Pipeline

```text
token shard -> teacher checkpoint -> full base loss -> token masking -> masked losses -> importance scores -> normalized scores -> importance shard
```

The generator uses `model.eval()` and `torch.no_grad()`. It does not train or modify the teacher checkpoint.

## 6. Smoke Test Result

The smoke test report is written to `docs/PREDICTIVE_IMPORTANCE_SMOKE_REPORT.md`.
It records the teacher checkpoint, token shard input directory, importance output directory, generated shard count, first shard shape, importance statistics, and pass flag.

Current Step 5 smoke result:

```text
PREDICTIVE_IMPORTANCE_SMOKE_PASS = true
generated importance shard count = 2
first importance shard shape = [8, 196]
importance output dir = /home/ubuntu22/tgpawb_world_model/data/importance_shards/dummy_toy_teacher
```

## 7. Pytest Result

Run:

```bash
/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python -m pytest tests -q
```

Current result:

```text
13 passed
```

## 8. Git Commit

Branch: `feature/tgpawb-step3-token-extraction`.
GitHub repo: `git@github.com:YangWang0709/world_model_yangwang.git`.
The exact pushed commit hash is recorded after commit in the local Step 5 summary and final handoff.
No PR is created for this step.

## 9. What Was Not Done

- no real V-JEPA / VideoMAE encoder
- no large model download
- no real dataset download
- no long teacher training
- no student training
- no VLM grounding
- no importance shard committed
- no checkpoint committed
- no token shard committed
- no password or token saved
- no PR created

## 10. Next Step Recommendation

Step 6 should train a student attention selector whose scores align with predictive importance while keeping future latent prediction close to the teacher.
The first version can use the dummy token shards plus generated importance shards for top-K or soft-mask selector training.
