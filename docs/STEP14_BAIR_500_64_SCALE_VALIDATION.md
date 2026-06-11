# STEP14 BAIR 500/64 Scale Validation

## 1. Goal

Scale the BAIR Robot Pushing small validation from 100/16 smoke artifacts to 500 train and 64 test clips while keeping the Step 13 selector choice fixed.

## 2. Why Stop Smoke Tuning

Step 13 selected `weighted_mse_alpha2` by downstream MSE on the 100/16 setting. Step 14 validates that choice at a larger scale instead of adding more small-sample loss variants.

## 3. Cloud Server Decision

No cloud server was required for this stage. The local RTX 5080 host is enough unless CUDA OOM persists after reducing batch size or token chunk size, in which case a 4090 / 48GB host becomes a reasonable next option.

## 4. Input

- BAIR TFDS dir: `/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset_500_64` source export from existing TFDS
- VideoMAE checkpoint: local `model_cache/videomae-base-finetuned-kinetics-local`
- train/test samples: `500` / `64`
- token shape: `[4, 392, 768]`
- topK: `16`

## 5. Pipeline

BAIR subset export -> VideoMAE tokens -> Teacher -> predictive importance -> weighted_mse_alpha2 selector -> StudentWorldModel -> baselines.

## 6. Configs

- `configs/bair_500_64_scale_validation.yaml`
- `configs/export_bair_subset_500_64.yaml`
- `configs/token_extraction_bair_videomae_500_64.yaml`
- `configs/train_teacher_bair_videomae_500_64.yaml`
- `configs/generate_importance_bair_videomae_500_64.yaml`
- `configs/train_selector_bair_videomae_500_64_weighted_mse.yaml`
- `configs/train_student_world_model_bair_videomae_500_64_weighted_mse.yaml`
- `configs/baseline_comparison_bair_videomae_500_64.yaml`

## 7. Results

- Teacher MSE: `1.4034927586714427`
- Student MSE: `1.6756061762571335`
- Student/Teacher ratio: `1.1938830221277916`
- learned vs random mean MSE delta: `-0.014098071389728073`
- learned vs uniform MSE delta: `-0.032474418481190925`
- learned vs teacher_importance_topk MSE delta: `0.13755552470684052`

See `docs/BAIR_500_64_SCALE_VALIDATION_REPORT.md` for the full table.

## 8. Sanity Gate

```json
{
  "checks": {
    "pipeline_outputs_present": true,
    "train_subset_count_500": true,
    "test_subset_count_64": true,
    "token_extraction_no_fallback": true,
    "token_shape_expected": true,
    "teacher_eval_mse_finite": true,
    "importance_stats_finite": true,
    "selector_metrics_finite": true,
    "student_metrics_finite": true,
    "baseline_summary_exists": true,
    "oom_false": true,
    "pytest_passed": true
  },
  "pass": true
}
```

## 9. Resource Usage

```json
{
  "before": {
    "disk_total_gib": 483.587,
    "disk_used_gib": 275.299,
    "disk_free_gib": 188.394,
    "ram_total_gib": 30.403,
    "ram_available_gib": 23.828,
    "ram_used_gib": 6.039,
    "torch_cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.461,
    "gpu_mem_used_gib": 0.932,
    "gpu_mem_free_gib": 14.529
  },
  "after": {
    "disk_total_gib": 483.587,
    "disk_used_gib": 279.29,
    "disk_free_gib": 184.402,
    "ram_total_gib": 30.403,
    "ram_available_gib": 18.931,
    "ram_used_gib": 10.921,
    "torch_cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.461,
    "gpu_mem_used_gib": 1.092,
    "gpu_mem_free_gib": 14.369
  },
  "elapsed_time_sec": 48.962,
  "max_ram_used_gib": 10.921,
  "max_gpu_mem_used_gib": 1.092,
  "gpu_mem_fraction": 0.07062932539939203,
  "oom": false,
  "cloud_recommendation": "not required for Step 14 local 500/64 validation"
}
```

## 10. Pytest

- command: `/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python -m pytest tests -q`
- passed: `true`

## 11. Git Commit

This document records the Step 14 code/report state. Commit hash is filled in the final local summary after push.

## 12. What Was Not Done

- no new model download
- no new dataset download
- no BAIR re-download
- no VideoMAE training
- no Step 11E/11F/12/13 overwrite
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

## 13. Next Step Recommendation

If the learned selector is stable against random and uniform, Step 15 can move to 1000/128 or multi-seed. If it only beats random, improve the compressor or StudentWorldModel before scaling further. If it is unstable, inspect Teacher importance quality.
