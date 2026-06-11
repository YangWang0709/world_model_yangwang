# BAIR 500/64 Scale Validation Report

Command: `python scripts/smoke_test_bair_500_64_scale_validation.py`

## Output Dirs

- subset: `/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset_500_64`
- tokens: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_500_64`
- importance: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_500_64`
- Teacher run: `/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_500_64_v1`
- Student run: `/home/ubuntu22/tgpawb_world_model/runs/student_world_model_bair_videomae_500_64_weighted_mse_v1`

## Unified Summary

# BAIR 500/64 Scale Validation Summary

- stage: `bair_500_64_scale_validation`
- train/test samples: `500` / `64`
- token shape: `[4, 392, 768]`
- topK: `16`
- token retention ratio: `0.040816326531`
- Teacher eval MSE: `1.40349276`
- Student future MSE: `1.67560618`
- Student/Teacher ratio: `1.193883`
- sanity gate pass: `true`
- cloud required: `false`

## Selector

- loss: `weighted_mse_alpha2`
- importance MSE: `0.04471610`
- Pearson corr: `0.152196`
- target topK overlap: `0.111328`
- selected teacher importance: `0.556490`

## Baselines

| policy | seed | student_future_mse | teacher_mse | ratio | topK overlap | selected importance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned_selector_weighted_mse_alpha2 | 0 | 1.64513243 | 1.40349276 | 1.172170 | 0.111328 | 0.556490 |
| random_k | 0 | 1.65309199 | 1.40349276 | 1.177841 | 0.044922 | 0.505603 |
| random_k | 1 | 1.65516942 | 1.40349276 | 1.179322 | 0.044922 | 0.521593 |
| random_k | 2 | 1.66943011 | 1.40349276 | 1.189483 | 0.038086 | 0.514990 |
| teacher_importance_topk | 0 | 1.50757691 | 1.40349276 | 1.074161 | 1.000000 | 0.873305 |
| uniform_k | 0 | 1.67760685 | 1.40349276 | 1.195309 | 0.055664 | 0.505338 |

## Comparisons

- learned vs random mean MSE delta: `-0.01409807`
- learned vs uniform MSE delta: `-0.03247442`
- learned vs teacher_importance_topk MSE delta: `0.13755552`
- learned selected importance vs random delta: `0.042428`
- learned selected importance vs uniform delta: `0.051151`

## Caveats

- none

## Sanity Gate

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

## Resource Summary

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

BAIR_500_64_SCALE_VALIDATION_PASS = true


## Pytest

```text
........................................................................ [ 68%]
.................................                                        [100%]
105 passed in 1.43s
```

BAIR_500_64_SCALE_VALIDATION_PASS = true
