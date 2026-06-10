# STEP4 Teacher Tiny Training Report

## 1. Goal

Step 4 implements a tiny full-token `TeacherWorldModel` training and overfit sanity check.

This stage uses the Step 3 dummy token shards only. It does not connect real V-JEPA or VideoMAE encoders, does not download large model weights, and does not run long training.

## 2. Files Added or Updated

- `.gitignore`
- `configs/train_teacher_dummy.yaml`
- `data/__init__.py`
- `data/token_shard_dataset.py`
- `models/teacher_world_model.py`
- `training/__init__.py`
- `training/teacher_trainer.py`
- `training/train_teacher.py`
- `eval/eval_teacher_prediction.py`
- `scripts/inspect_teacher_checkpoint.py`
- `scripts/smoke_test_teacher_training.py`
- `docs/TEACHER_TRAINING_SMOKE_REPORT.md`
- `docs/STEP4_TEACHER_TINY_TRAINING.md`
- `tests/test_token_shard_dataset.py`
- `tests/test_teacher_training_step.py`
- `tests/test_teacher_checkpoint.py`

## 3. TokenShardDataset

`TokenShardDataset` reads Step 3 token shards from a shard file, shard directory, or shard file list. The default Step 4 input directory is:

`/home/ubuntu22/tgpawb_world_model/data/token_shards/dummy_toy`

It uses `load_token_shard`, `validate_token_shard`, and `summarize_token_shard` from `data/token_shards.py`. In this tiny setting, shards are loaded into memory for simple, explicit sanity training.

Each item returns:

- `past_tokens`: `[N, D]`
- `future_tokens`: `[N, D]` or `[D]`
- `sample_id`
- `task_text`
- `metadata`
- `shard_path`

The `token_shard_collate_fn` returns batched `past_tokens` as `[B, N, D]` and `future_tokens` as `[B, N, D]` or `[B, D]`.

## 4. TeacherWorldModel

The teacher input and output contract is:

- input: `[B, N, D]`
- output: `[B, D]`

The current lightweight teacher uses mean pooling over tokens followed by `LayerNorm` and a small GELU MLP.

Default config:

- `token_dim`: `768`
- `hidden_dim`: `512`
- `output_dim`: `768`
- `num_layers`: `2`
- `dropout`: `0.0`
- `pool`: `mean`

## 5. Training Pipeline

Pipeline:

`token shard dataset -> DataLoader -> TeacherWorldModel -> future target pooling -> MSE loss -> AdamW -> metrics/checkpoint`

If `future_tokens` is `[B, N, D]`, the target is mean pooled over `N` to `[B, D]`. If it is already `[B, D]`, it is used directly. Prediction and target shapes must match exactly.

Training outputs are written under ignored `runs/`:

- `runs/teacher_dummy_tiny_v1/metrics.jsonl`
- `runs/teacher_dummy_tiny_v1/summary.json`
- `runs/teacher_dummy_tiny_v1/eval_summary.json`
- `runs/teacher_dummy_tiny_v1/checkpoints/teacher_world_model_step_000030.pt`

## 6. Smoke Test Result

Smoke report: `docs/TEACHER_TRAINING_SMOKE_REPORT.md`

- run dir: `/home/ubuntu22/tgpawb_world_model/runs/teacher_dummy_tiny_v1`
- num steps: `30`
- initial loss: `0.03960206359624863`
- final loss: `9.57007723627612e-05`
- best loss: `9.57007723627612e-05`
- loss_decreased: `true`
- device: `cuda`
- eval mse: `8.54052177601261e-05`
- checkpoint path: `/home/ubuntu22/tgpawb_world_model/runs/teacher_dummy_tiny_v1/checkpoints/teacher_world_model_step_000030.pt`
- `TEACHER_TRAINING_SMOKE_PASS = true`

## 7. Pytest Result

All tests passed in the remote `env_isaaclab` conda environment:

`9 passed`

This includes the original Step 2 and Step 3 tests plus the new Step 4 token shard dataset, teacher training step, and checkpoint tests.

## 8. Git Commit

- Repository: `git@github.com:YangWang0709/world_model_yangwang.git`
- Branch: `feature/tgpawb-step3-token-extraction`
- Commit message: `feat(tgpawb): add teacher tiny training`
- Push target: `origin/feature/tgpawb-step3-token-extraction`
- PR: not created

The exact commit hash and push status are recorded in the local Step 4 summary and final response after commit/push.

## 9. What Was Not Done

- No real V-JEPA / VideoMAE encoder.
- No large model download.
- No real dataset download.
- No long training.
- No student training.
- No predictive importance generation yet.
- No VLM grounding.
- No checkpoint committed.
- No token shard committed.
- No toy dataset committed.
- No password or token saved.
- No PR created.

## 10. Next Step Recommendation

Recommended Step 5: implement offline predictive token importance generation based on the trained teacher checkpoint. The first version can use block masking or token masking on dummy shards:

`I_i = Loss(Teacher(z_without_i), z_future) - Loss(Teacher(z_full), z_future)`

Save generated importance shards as local artifacts, but do not commit generated `.pt` files.

