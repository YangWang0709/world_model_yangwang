# STEP7 Student World Model Training Report

## 1. Goal

Step 7 trains a compressed Student world-model chain on `structured_toy`: `past_tokens -> frozen Step6 AttentionSelector -> topK selected tokens -> TokenCompressor -> StudentWorldModel -> predicted future latent`.

## 2. Default Training Scope

The Step6 selector is frozen by default. Only `TokenCompressor` and `StudentWorldModel` are optimized unless the config explicitly sets `selector.frozen: false`.

## 3. Target

`future_tokens` are converted to a `[B, D]` target with the same helper used by teacher training. Rank-3 future tokens are mean-pooled over the token dimension; rank-2 targets are used directly.

## 4. Files Added or Updated

- `configs/train_student_world_model_structured_toy.yaml`
- `models/token_compressor.py`
- `models/student_world_model.py`
- `training/student_world_model_trainer.py`
- `training/train_student_world_model.py`
- `eval/eval_student_world_model.py`
- `eval/eval_teacher_student_gap.py`
- `scripts/smoke_test_student_world_model_training.py`
- `scripts/inspect_student_world_model_checkpoint.py`
- `tests/test_student_world_model_training_step.py`
- `tests/test_student_world_model_eval.py`
- `tests/test_teacher_student_gap.py`

## 5. Smoke Test Result

- run dir: `/home/ubuntu22/tgpawb_world_model/runs/student_world_model_structured_toy_v1`
- checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/student_world_model_structured_toy_v1/checkpoints/student_world_model_step_000300.pt`
- selector checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_structured_toy_v1/checkpoints/student_selector_step_000200.pt`
- selector_frozen: `True`
- num steps: `300`
- initial loss: `0.4130794405937195`
- final loss: `0.002919309539720416`
- best loss: `0.002136853989213705`
- loss_decreased: `True`
- selected_top1_hit_rate: `1.0`
- selected_topk_hit_rate: `1.0`
- selected_key_coverage: `1.0`
- token_retention_ratio: `0.02040816326530612`
- STUDENT_WORLD_MODEL_TRAINING_SMOKE_PASS: `true`

## 6. Teacher/Student Future Prediction Gap

- teacher_future_mse: `0.002693264357124766`
- student_future_mse: `0.0023518727781871953`
- student_teacher_gap: `-0.00034139157893757054`
- student_teacher_ratio: `0.8732424546315134`

## 7. Pytest Result

`/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python -m pytest tests -q` completed with `26 passed in 0.97s`.

The regression smoke sequence also passed after the Step 7 changes: minimal pipeline, token extraction, teacher training, predictive importance, structured importance signal quality, Student selector training, and Student world-model training.

## 8. Git Commit

Branch: `feature/tgpawb-step3-token-extraction`. Commit hash and push status are recorded after commit/push. No PR is created in this stage.

## 9. What Was Not Done

- no real dataset download
- no real V-JEPA / VideoMAE encoder
- no large model download
- no VLM grounding
- no RL / policy optimization
- no generated `.pt` committed
- no checkpoint committed
- no password or token saved
- no PR created

## 10. Next Step Recommendation

Move from the controlled `structured_toy` setup to a slightly less synthetic offline token set only after the compressed Student chain keeps stable coverage and finite teacher/student gap metrics.
