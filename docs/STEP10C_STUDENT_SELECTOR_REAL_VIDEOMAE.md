# STEP10C Student Selector on Real VideoMAE Importance

## 1. Goal

Step 10C trains a Student Attention Selector smoke model on the real VideoMAE predictive-importance labels generated in Step 10B.

The selector consumes real VideoMAE `past_tokens [B, 784, 768]`, predicts token logits `[B, 784]`, applies sigmoid probabilities, and regresses them to `importance_scores_norm [B, 784]`.

This is an 8-sample smoke validation, not a paper-scale result.

## 2. Cloud Server Decision

No cloud server was required for this stage. Step 10C did not run the VideoMAE encoder, did not retrain the Teacher, and did not train a StudentWorldModel. It only read 8 existing token samples and 8 existing importance samples on the local RTX 5080 server.

The smoke completed without OOM. GPU memory moved from 0.948 GiB to 1.054 GiB used, and RAM moved from 4.202 GiB to 5.012 GiB used. A 4090 / 48GB server is not needed for this bounded smoke, but it is recommended before scaling to larger real datasets.

## 3. Input

- token shard dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke`
- importance shard dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke`
- samples: `8`
- token shape: `[B, 784, 768]`
- importance shape: `[B, 784]`
- source encoder: `videomae`
- used fallback: `false`

## 4. Training Setup

- config: `configs/train_student_selector_real_video_videomae_smoke.yaml`
- model: `AttentionSelector`
- token_dim: `768`
- hidden_dim: `256`
- target: `importance_scores_norm`
- loss: MSE importance regression
- key_token_mask: not required
- ranking loss: disabled
- batch_size: `2`
- max_steps: `300`
- topk: `16`

Because the real minimal samples do not include structured toy `key_token_mask` labels, Step 10C does not report key-token hit rate. It evaluates selector quality with importance MSE/MAE, Pearson correlation, Teacher top-1 overlap, Teacher top-k overlap, and the mean Teacher importance of selected tokens.

`target_top1_overlap` is defined as `pred_top1 == target_top1`. `target_topk_overlap` is `intersection(pred_topK, target_topK) / K`.

## 5. Smoke Test Result

Command:

```bash
/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_student_selector_real_video_videomae.py
```

Result:

- `STUDENT_SELECTOR_REAL_VIDEOMAE_SMOKE_PASS = true`
- num_steps: `300`
- initial loss: `0.09144660085439682`
- final loss: `0.01454525999724865`
- best loss: `0.006556759588420391`
- loss_decreased: `true`
- importance_mse: `0.018842598423361778`
- importance_mae: `0.10066770762205124`
- pearson_corr_mean: `0.5616448521614075`
- target_top1_overlap: `0.125`
- target_topk_overlap: `0.3125`
- selected_teacher_importance_mean: `0.6910883784294128`
- random_teacher_importance_mean: `0.46076127886772156`
- checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/checkpoints/student_selector_step_000300.pt`
- OOM: `false`

The full smoke report is in `docs/STUDENT_SELECTOR_REAL_VIDEOMAE_SMOKE_REPORT.md`.

## 6. Pytest Result

Full pytest was run after the Step 10C smoke and eval:

```text
52 passed in 1.23s
```

The new tests cover the real VideoMAE Student selector config bounds and the no-key-mask importance metrics, including a constant-row Pearson correlation case that returns zero instead of NaN.

## 7. Git Commit

- branch: `feature/tgpawb-step3-token-extraction`
- repository: `git@github.com:YangWang0709/world_model_yangwang.git`
- PR: not created
- main merge: not performed

The exact commit hash is recorded after commit/push in the local Step 10C summary and final handoff.

## 8. What Was Not Done

- no new model download
- no new real dataset download
- no VideoMAE encoder rerun
- no VideoMAE training
- no Teacher retraining
- no StudentWorldModel training
- no TokenCompressor connection
- no VLM grounding
- no RL / policy optimization
- no large dependency installation
- no model weight committed
- no generated token shard `.pt` committed
- no generated importance shard `.pt` committed
- no checkpoint committed
- no password or token saved
- no PR created

## 9. Next Step Recommendation

Step 10D can connect the real VideoMAE selector to TokenCompressor and StudentWorldModel, then train a compressed Student future-prediction smoke. That should still remain small and explicit about being a smoke test.

After that, Step 11 can introduce the first approved public real dataset subset, such as a small BAIR Robot Pushing or ManiSkill subset, with resource planning revisited before scaling extraction, importance generation, or selector training.
