# STEP16 BAIR Downstream Utilization / Compressor Ablation

## 1. Goal

Diagnose the Step15 gap where the learned selector selects higher Teacher-importance tokens but the downstream StudentWorldModel does not reliably beat uniform selection.

## 2. Motivation

- learned selected importance > random/uniform
- learned MSE is only slightly better than random
- learned MSE is worse than uniform
- the likely bottleneck is selected-token utilization in the compressor / StudentWorldModel.

## 3. Cloud Server Decision

No cloud server is required for this stage because it reuses Step15 BAIR 1000/128 tokens, importance shards, Teacher checkpoint, and selector checkpoints. A larger 4090 / 48GB server is only recommended if CUDA OOM persists after reducing batch size.

## 4. Input

- train_token_shard_dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_1000_128/train`
- test_token_shard_dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_1000_128/test`
- train_importance_shard_dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_1000_128/train`
- test_importance_shard_dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_1000_128/test`
- teacher_checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_1000_128_v1/checkpoints/teacher_world_model_step_001200.pt`
- selector_root: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_1000_128_weighted_mse_multiseed_v1`
- baseline_aggregate: `/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_1000_128_multiseed_v1/baseline_multiseed_aggregate.json`

## 5. Variants

- current_perceiver_like
- mean_pool_selected
- attention_pool_selected
- selector_score_weighted_pool
- transformer_encoder_selected
- cross_attention_latent_bottleneck
- hybrid_learned8_uniform8_perceiver
- hybrid_learned8_uniform8_cross_attention

## 6. Phase A

# Step16 Phase A Summary

| variant | seed | success | student_future_mse | ratio | selected_importance | topK_overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| attention_pool_selected | 0 | true | 1.56332328 | 1.260268 | 0.540891 | 0.110352 |
| cross_attention_latent_bottleneck | 0 | true | 1.62373509 | 1.308969 | 0.540891 | 0.110352 |
| current_perceiver_like | 0 | true | 1.59231013 | 1.283636 | 0.540891 | 0.110352 |
| hybrid_learned8_uniform8_cross_attention | 0 | true | 1.61138300 | 1.299011 | 0.519385 | 0.089844 |
| hybrid_learned8_uniform8_perceiver | 0 | true | 1.61423208 | 1.301308 | 0.519385 | 0.089844 |
| mean_pool_selected | 0 | true | 1.42345227 | 1.147512 | 0.540891 | 0.110352 |
| selector_score_weighted_pool | 0 | true | 1.42233196 | 1.146608 | 0.540891 | 0.110352 |
| transformer_encoder_selected | 0 | true | 1.62044428 | 1.306316 | 0.540891 | 0.110352 |


## 7. Phase B

# Step16 Phase B Summary

## Aggregate

| variant | seeds | n | mse_mean | mse_std | ratio_mean | ratio_std | selected_importance_mean | selected_importance_std | topK_overlap_mean | topK_overlap_std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_pool_selected | 0,1,2 | 3 | 1.39506111 | 0.00904936 | 1.124624 | 0.007295 | 0.547769 | 0.008320 | 0.115397 | 0.003618 |
| selector_score_weighted_pool | 0,1,2 | 3 | 1.39673269 | 0.00993852 | 1.125972 | 0.008012 | 0.547769 | 0.008320 | 0.115397 | 0.003618 |

## Rows

| variant | seed | student_future_mse | ratio | selected_importance | topK_overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_pool_selected | 0 | 1.40779206 | 1.134887 | 0.540891 | 0.110352 |
| mean_pool_selected | 1 | 1.38756485 | 1.118581 | 0.542941 | 0.118652 |
| mean_pool_selected | 2 | 1.38982642 | 1.120404 | 0.559475 | 0.117188 |
| selector_score_weighted_pool | 0 | 1.41066506 | 1.137203 | 0.540891 | 0.110352 |
| selector_score_weighted_pool | 1 | 1.38816080 | 1.119062 | 0.542941 | 0.118652 |
| selector_score_weighted_pool | 2 | 1.39137223 | 1.121650 | 0.559475 | 0.117188 |


## 8. Comparison vs Step15

- best Step16 vs Step15 learned delta: `-0.17817141`
- best Step16 vs Step15 random delta: `-0.18338907`
- best Step16 vs Step15 uniform delta: `-0.17296126`
- best Step16 vs teacher_importance_topk delta: `-0.12955184`
- hybrid reached Phase B: `false`
- cross-attention reached Phase B: `false`

## 9. Interpretation

If hybrid learned+uniform is best, the learned tokens likely need additional global coverage. If cross-attention is best, the current compressor is likely too weak for non-uniform selected tokens. If no variant beats uniform, downstream architecture alone is not enough and Step17 should prioritize action-conditioned dynamics.

## 10. Sanity Gate

```json
{
  "checks": {
    "phase_a_at_least_6_success": true,
    "phase_b_at_least_2_variants": true,
    "all_success_metrics_finite": true,
    "oom_false": true,
    "pytest_passed": true,
    "env_isaaclab_unpolluted": true
  },
  "pass": true
}
```

## 11. Resource Usage

```json
{
  "before": {
    "disk_total_gib": 483.587,
    "disk_used_gib": 287.331,
    "disk_free_gib": 176.361,
    "ram_total_gib": 30.403,
    "ram_available_gib": 24.987,
    "ram_used_gib": 5.002,
    "torch_cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.461,
    "gpu_mem_used_gib": 0.966,
    "gpu_mem_free_gib": 14.495
  },
  "after": {
    "disk_total_gib": 483.587,
    "disk_used_gib": 287.702,
    "disk_free_gib": 175.991,
    "ram_total_gib": 30.403,
    "ram_available_gib": 21.081,
    "ram_used_gib": 8.909,
    "torch_cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.461,
    "gpu_mem_used_gib": 1.214,
    "gpu_mem_free_gib": 14.246
  },
  "elapsed_time_sec": 42.842,
  "max_ram_used_gib": 8.909,
  "max_gpu_mem_used_gib": 1.214,
  "gpu_mem_fraction": 0.07852014746782227,
  "oom": false,
  "cloud_recommendation": "not required for Step 16 local downstream utilization ablation"
}
```

## 12. Pytest

- command: `/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python -m pytest tests -q`
- passed: `true`

## 13. Git Commit

Commit hash is recorded in the final local summary after push.

## 14. What Was Not Done

- no new model download
- no new dataset download
- no BAIR re-download
- no VideoMAE extraction
- no VideoMAE training
- no Teacher retraining
- no importance regeneration
- no selector retraining
- no Step11-15 overwrite
- no VLM grounding
- no RL
- no action-conditioned world model
- no model weight committed
- no BAIR data committed
- no token shard committed
- no importance shard committed
- no checkpoint committed
- no password/token saved
- no PR created

## 15. Next Step Recommendation

Use the best Step16 utilizer if it beats Step15 uniform. If it does not, proceed to an action-conditioned StudentWorldModel in Step17.
