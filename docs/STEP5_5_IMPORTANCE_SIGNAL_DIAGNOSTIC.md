# STEP5.5 Importance Signal Diagnostic Report

## 1. Why Step 5.5 Was Needed

Step 5 passed the predictive-importance pipeline smoke test, but the dummy toy setup produced near-all-negative raw importance and all-zero normalized importance. That means the pipeline works mechanically, while the toy signal is not suitable yet for training a Student selector.

## 2. Root Cause Analysis

- Dummy encoder tokens are weakly related or unrelated to visual content.
- Mean pooling dilutes the effect of one token across 196 tokens.
- Zero masking can reduce noise instead of increasing prediction loss.
- The dummy future target lacks a clear causal dependency on local past tokens.
- The tiny teacher objective can be too easy, so single-token occlusion has a very small effect.

## 3. Structured Token Toy Setup

- num_samples: `32`
- num_tokens: `196`
- token_dim: `768`
- num_key_tokens: `4`
- signal_scale: `3.0`
- noise_scale: `0.05`
- key labels: `metadata.key_token_indices` and optional `aux_labels.key_token_mask`

## 4. Teacher Training

- run dir: `/home/ubuntu22/tgpawb_world_model/runs/teacher_structured_toy_tiny_v1`
- checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/teacher_structured_toy_tiny_v1/checkpoints/teacher_world_model_step_000100.pt`
- num steps: `100`
- initial loss: `0.3507133722305298`
- final loss: `0.003032910404726863`
- best loss: `0.00254505081102252`
- loss_decreased: `True`

## 5. Importance Generation

- output dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/structured_toy_teacher`
- generated shards: `4`
- importance mean/std/min/max: `0.0005309324478730559 / 0.00401536887511611 / -2.428889274597168e-06 / 0.0474613681435585`
- normalized importance mean/std/min/max: `0.013094807043671608 / 0.09739396721124649 / 0.0 / 1.0`

## 6. Importance Quality Evaluation

- positive_importance_ratio: `0.6173469424247742`
- key_token_importance_mean: `0.025051988661289215`
- non_key_token_importance_mean: `1.7984150701977342e-07`
- key_vs_non_key_gap: `0.025051808819782195`
- top1_hit_rate: `1.0`
- topk_hit_rate: `1.0`
- pass: `True`

## 7. Smoke Test Result

`IMPORTANCE_SIGNAL_QUALITY_PASS = true`. See `docs/IMPORTANCE_SIGNAL_QUALITY_REPORT.md`.

## 8. Pytest Result

Post-smoke validation command:

`/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python -m pytest tests -q`

Observed result after Step 5.5 changes: `17 passed in 0.94s`.

## 9. Git Commit

Branch: `feature/tgpawb-step3-token-extraction`. Commit hash and push status are recorded after git commit/push. No PR is created in this stage.

## 10. What Was Not Done

- no real dataset download
- no real V-JEPA / VideoMAE encoder
- no large model download
- no long training
- no student training
- no VLM grounding
- no generated `.pt` committed
- no checkpoint committed
- no password or token saved
- no PR created

## 11. Next Step Recommendation

Step 6 should train a Student attention selector against the structured toy importance labels first, verifying that it can recover key tokens before moving to less controlled data.
