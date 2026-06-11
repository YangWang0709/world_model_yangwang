# STEP11F Student World Model on BAIR VideoMAE Tokens

## 1. Goal

Step 11F connects the BAIR Student selector, `TokenCompressor`, and `StudentWorldModel` into a compressed future-prediction smoke pipeline on public BAIR Robot Pushing small clips represented as VideoMAE tokens.

## 2. Cloud Server Decision

No cloud server is required for this stage. The run reads existing BAIR VideoMAE token shards and predictive-importance shards, and it does not run the VideoMAE encoder.

- OOM: `False`
- max RAM used GiB: `7.369956970214844`
- max GPU memory used GiB: `1.2109375`
- cloud recommendation: `not required for Step 11F smoke`

## 3. Input

- train token dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train`
- test token dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test`
- train importance dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/train`
- test importance dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/test`
- selector checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/checkpoints/student_selector_step_000100.pt`
- teacher checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1/checkpoints/teacher_world_model_step_000300.pt`
- train samples: `100`
- test samples: `16`
- token shape: `[B, 392, 768]`
- source encoder: `videomae`

## 4. Pipeline

`past_tokens -> frozen selector -> topK selected tokens -> TokenCompressor -> StudentWorldModel -> future latent prediction`.

## 5. Training Setup

- selector frozen: `True`
- topK: `16`
- token_retention_ratio: `0.04081632653061224`
- compressed latents: `16`
- target: mean-pooled `future_tokens` latent
- optimized modules: `TokenCompressor` + `StudentWorldModel`
- eval split: BAIR test

## 6. Smoke Test Result

- initial_loss: `12.400135040283203`
- final_loss: `1.508751630783081`
- best_loss: `1.0455163717269897`
- loss_decreased: `True`
- student_future_mse: `1.96855632464091`
- selector_target_top1_overlap: `0.0`
- selector_target_topK_overlap: `0.1640625`
- selected_teacher_importance_mean: `0.5927466750144958`
- STUDENT_WORLD_MODEL_BAIR_VIDEOMAE_SMOKE_PASS: `true`

## 7. Teacher-Student Gap

- teacher_mse: `1.686881383260091`
- student_mse: `1.96855632464091`
- student_teacher_gap: `0.28167494138081883`
- student_teacher_ratio: `1.1669796964837265`
- token_retention_ratio: `0.04081632653061224`

## 8. Pytest Result

`79 passed in 1.41s`

## 9. Git Commit

Branch: `feature/tgpawb-step3-token-extraction`. The final commit hash and push status are recorded after commit/push in `STEP11F_LOCAL_SUMMARY.md` and the final response. No PR is created.

## 10. What Was Not Done

- no new model download
- no new dataset download
- no BAIR re-download
- no VideoMAE training
- no Teacher retraining
- no selector retraining
- no VLM grounding
- no RL / policy optimization
- no model weight committed
- no BAIR data committed
- no token shard committed
- no importance shard committed
- no checkpoint committed
- no password or token saved
- no PR created

## 11. Next Step Recommendation

Step 12 should run a bounded BAIR baseline comparison: Random-K Student, Uniform-K Student, learned-selector Student, Teacher reference, and optionally an oracle-like top-importance Student. Compare token retention, future MSE, teacher-student ratio, and selected teacher importance.
