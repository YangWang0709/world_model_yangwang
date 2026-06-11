# STEP10D Student World Model on Real VideoMAE Tokens

## 1. Goal

Step 10D connects the real VideoMAE Student selector, `TokenCompressor`, and `StudentWorldModel` into a compressed future-prediction smoke pipeline.

## 2. Cloud Server Decision

No cloud server is required for this stage because the run only reads 8 existing real VideoMAE token samples and does not run the VideoMAE encoder. If future dataset scale grows to hundreds or thousands of clips, a 4090 / 48GB host should be reconsidered.

- OOM: `False`
- max RAM used GiB: `4.6351776123046875`
- max GPU memory used GiB: `1.23046875`

## 3. Input

- token_shard_dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke`
- importance_shard_dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke`
- selector checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/checkpoints/student_selector_step_000300.pt`
- teacher checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/teacher_real_video_videomae_smoke_v1/checkpoints/teacher_world_model_step_000100.pt`
- num_samples: `8`
- token shape: `[B, 784, 768]`
- source encoder: `videomae`

## 4. Pipeline

`past_tokens -> frozen selector -> topK selected tokens -> TokenCompressor -> StudentWorldModel -> future latent prediction`.

## 5. Training Setup

- selector_frozen: `True`
- topK: `16`
- token_retention_ratio: `0.02040816326530612`
- compressed latents: `16`
- target: mean-pooled `future_tokens` latent
- optimized modules: `TokenCompressor` + `StudentWorldModel`

## 6. Smoke Test Result

- initial_loss: `23.0651912689209`
- final_loss: `0.04984317347407341`
- best_loss: `0.039038270711898804`
- loss_decreased: `True`
- student_future_mse: `0.04321051016449928`
- selector_target_top1_overlap: `0.125`
- selector_target_topK_overlap: `0.3125`
- selected_teacher_importance_mean: `0.6910883784294128`
- STUDENT_WORLD_MODEL_REAL_VIDEOMAE_SMOKE_PASS: `true`

## 7. Teacher-Student Gap

- teacher_mse: `0.166172057390213`
- student_mse: `0.04321051016449928`
- student_teacher_gap: `-0.12296154722571373`
- student_teacher_ratio: `0.2600347545979427`
- token_retention_ratio: `0.02040816326530612`

## 8. Pytest Result

`/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python -m pytest tests -q` passed with `56 passed in 1.27s`.

## 9. Git Commit

Branch: `feature/tgpawb-step3-token-extraction`. The final commit hash and push status are recorded after commit/push in `STEP10D_LOCAL_SUMMARY.md` and the final response. No PR is created.

## 10. What Was Not Done

- no new model download
- no new real dataset download
- no VideoMAE training
- no Teacher retraining
- no selector retraining
- no VLM grounding
- no RL / policy optimization
- no model weight committed
- no generated `.pt` committed
- no checkpoint committed
- no password or token saved
- no PR created

## 11. Next Step Recommendation

Step 11 should start with a tiny public real-dataset subset, such as BAIR Robot Pushing or a small ManiSkill/RLBench task, and first build dataset indexing, loading, and token extraction before any larger-scale training.
