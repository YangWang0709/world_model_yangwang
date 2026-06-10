# STEP6 Student Selector Training Report

## 1. Goal

Step 6 trains only the Student Attention Selector to predict Step 5.5 `structured_toy` normalized predictive-importance labels. This stage does not connect real V-JEPA / VideoMAE, VLM grounding, or RL.

## 2. Why Step 6 Is Now Valid

Step 5.5 produced positive structured importance signal: key token importance is higher than non-key token importance, normalized importance is not all zero, and structured toy top-k recovery reached 1.0.

## 3. Files Added or Updated

- `data/student_selector_dataset.py`
- `training/student_selector_trainer.py`
- `training/train_student_selector.py`
- `eval/eval_student_selector.py`
- `scripts/smoke_test_student_selector_training.py`
- `scripts/inspect_student_selector_checkpoint.py`
- `configs/train_student_selector_structured_toy.yaml`
- `tests/test_student_selector_dataset.py`
- `tests/test_student_selector_training_step.py`
- `tests/test_student_selector_eval.py`

## 4. StudentSelectorDataset

`StudentSelectorDataset` aligns token shards and importance shards by `sample_id`, returning `past_tokens`, normalized importance labels, raw importance scores, key token masks, task text, and metadata.

## 5. AttentionSelector

The selector consumes `[B, N, D]` tokens and emits `[B, N]` logits. Training applies `sigmoid(logits)` and regresses probabilities to `importance_scores_norm`.

## 6. Training Pipeline

`structured_toy` token shards plus importance shards feed the paired dataset. The selector is trained with importance regression loss plus a small key/non-key ranking margin loss. Metrics, summary, and checkpoint are written under ignored `runs/`.

## 7. Smoke Test Result

- run dir: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_structured_toy_v1`
- num steps: `200`
- initial loss: `0.26841574907302856`
- final loss: `0.000204412717721425`
- best loss: `0.000204412717721425`
- loss_decreased: `True`
- key_score_mean: `0.6109598278999329`
- non_key_score_mean: `0.010969343595206738`
- key_vs_non_key_gap: `0.5999904843047261`
- top1_hit_rate: `1.0`
- topk_hit_rate: `1.0`
- STUDENT_SELECTOR_TRAINING_SMOKE_PASS: `true`

## 8. Eval Result

- importance_mse: `0.00022361590526998043`
- key_score_mean: `0.6109598278999329`
- non_key_score_mean: `0.010969343595206738`
- key_vs_non_key_gap: `0.5999904843047261`
- top1_hit_rate: `1.0`
- topk_hit_rate: `1.0`

## 9. Pytest Result

Post-change validation command:

`/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python -m pytest tests -q`

Observed result after Step 6 changes: `20 passed in 0.97s`.

## 10. Git Commit

Branch: `feature/tgpawb-step3-token-extraction`. Commit hash and push status are recorded after commit/push. No PR is created in this stage.

## 11. What Was Not Done

- no real dataset download
- no real V-JEPA / VideoMAE encoder
- no large model download
- no VLM grounding
- no StudentWorldModel full training yet
- no token compressor training yet
- no RL / policy optimization
- no generated `.pt` committed
- no checkpoint committed
- no password or token saved
- no PR created

## 12. Next Step Recommendation

Step 7 should connect the selector to the token compressor and StudentWorldModel on `structured_toy`, validating compressed Student future-latent prediction under a low token budget.
