# STEP15 BAIR 1000/128 Multi-Seed Validation

## 1. Goal

Scale the BAIR Robot Pushing small validation from Step 14's 500/64 setting to 1000 train clips and 128 test clips, then evaluate stability across multiple seeds.

## 2. Why Multi-Seed Now

Step 14 fixed the selector choice to `weighted_mse_alpha2` and showed it beating random and uniform on 500/64. Step 15 tests whether that result is stable rather than tuning more small-sample variants.

## 3. Cloud Server Decision

No cloud server is required when the local RTX 5080 run stays under the disk/RAM/GPU limits. A 4090 / 48GB host is only a recommendation if CUDA OOM persists after reducing batch size or token chunk size.

## 4. Input

- BAIR subset path: `/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset_1000_128`
- VideoMAE checkpoint: local `model_cache/videomae-base-finetuned-kinetics-local`
- train/test samples: `1000` / `128`
- token shape: `[4, 392, 768]`
- topK: `16`
- seeds: `[0, 1, 2]`

## 5. Pipeline

BAIR subset export -> VideoMAE tokens -> Teacher -> predictive importance -> weighted_mse_alpha2 selector multi-seed -> StudentWorldModel multi-seed -> baseline comparison multi-seed.

## 6. Configs

- `configs/bair_1000_128_multiseed_validation.yaml`
- `configs/export_bair_subset_1000_128.yaml`
- `configs/token_extraction_bair_videomae_1000_128.yaml`
- `configs/train_teacher_bair_videomae_1000_128.yaml`
- `configs/generate_importance_bair_videomae_1000_128.yaml`
- `configs/train_selector_bair_videomae_1000_128_weighted_mse.yaml`
- `configs/train_student_world_model_bair_videomae_1000_128_weighted_mse.yaml`
- `configs/baseline_comparison_bair_videomae_1000_128_multiseed.yaml`

## 7. Results

- Teacher MSE: `1.240468766540289`
- learned MSE mean/std: `1.5732325199577544` / `0.02413252123643708`
- random MSE mean/std: `1.5784501764509413` / `0.037072805101552024`
- uniform MSE: `1.568022367854913`
- teacher_importance_topk MSE: `1.5246129507819812`
- learned vs random mean MSE delta: `-0.005217656493186951`
- learned vs uniform MSE delta: `0.00521015210284137`
- learned vs teacher_importance_topk MSE delta: `0.04861956917577315`

See `docs/BAIR_1000_128_MULTI_SEED_VALIDATION_REPORT.md` for the full tables.

## 8. Sanity Gate

```json
{
  "checks": {
    "pipeline_outputs_present": true,
    "train_subset_count_1000": true,
    "test_subset_count_128": true,
    "token_extraction_no_fallback": true,
    "token_shape_expected": true,
    "teacher_eval_mse_finite": true,
    "importance_stats_finite": true,
    "selector_multiseed_metrics_finite": true,
    "student_multiseed_metrics_finite": true,
    "baseline_multiseed_summary_exists": true,
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
    "disk_used_gib": 279.313,
    "disk_free_gib": 184.38,
    "ram_total_gib": 30.403,
    "ram_available_gib": 24.989,
    "ram_used_gib": 4.965,
    "torch_cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.461,
    "gpu_mem_used_gib": 1.006,
    "gpu_mem_free_gib": 14.455
  },
  "after": {
    "disk_total_gib": 483.587,
    "disk_used_gib": 287.323,
    "disk_free_gib": 176.369,
    "ram_total_gib": 30.403,
    "ram_available_gib": 17.834,
    "ram_used_gib": 12.117,
    "torch_cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.461,
    "gpu_mem_used_gib": 1.145,
    "gpu_mem_free_gib": 14.316
  },
  "elapsed_time_sec": 96.901,
  "max_ram_used_gib": 12.117,
  "max_gpu_mem_used_gib": 1.145,
  "gpu_mem_fraction": 0.07405730547830024,
  "oom": false,
  "cloud_recommendation": "not required for Step 15 local 1000/128 multi-seed validation"
}
```

## 10. Pytest

- command: `/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python -m pytest tests -q`
- passed: `true`

## 11. Git Commit

This document records the Step 15 code/report state. Commit hash is filled in the final local summary after push.

## 12. What Was Not Done

- no new model download
- no new dataset download
- no BAIR re-download
- no VideoMAE training
- no Step 11E/11F/12/13/14 overwrite
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

If learned weighted_mse_alpha2 remains stable against random and uniform, Step 16 can either scale again or start the next modeling upgrade. If it only beats random, improve the compressor/StudentWorldModel first. If it is unstable, diagnose Teacher importance quality and seed control.
