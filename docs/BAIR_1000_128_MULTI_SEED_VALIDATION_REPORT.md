# BAIR 1000/128 Multi-Seed Validation Report

Command: `python scripts/smoke_test_bair_1000_128_multiseed_validation.py`

## Output Dirs

- subset: `/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset_1000_128`
- tokens: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_1000_128`
- importance: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_1000_128`
- Teacher run: `/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_1000_128_v1`
- selector multi-seed run: `runs/student_selector_bair_videomae_1000_128_weighted_mse_multiseed_v1`
- StudentWorldModel multi-seed run: `runs/student_world_model_bair_videomae_1000_128_weighted_mse_multiseed_v1`
- baseline multi-seed run: `runs/baseline_comparison_bair_videomae_1000_128_multiseed_v1`

## Unified Summary

# BAIR 1000/128 Multi-Seed Validation Summary

- stage: `bair_1000_128_multiseed_validation`
- train/test samples: `1000` / `128`
- token shape: `[4, 392, 768]`
- topK: `16`
- token retention ratio: `0.040816326531`
- Teacher eval MSE: `1.24046877`
- selector loss: `weighted_mse_alpha2`
- selector seeds: `[0, 1, 2]`
- sanity gate pass: `true`
- cloud required: `false`

## Selector Multi-Seed

| seed | importance_mse | topK_overlap | selected_importance |
| ---: | ---: | ---: | ---: |
| 0 | 0.04172986 | 0.110352 | 0.540891 |
| 1 | 0.04059245 | 0.118652 | 0.542941 |
| 2 | 0.04038057 | 0.117188 | 0.559475 |

## StudentWorldModel Multi-Seed

| seed | student_future_mse | teacher_mse | ratio | selected_importance |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.60635101 | 1.24046873 | 1.294955 | 0.540891 |
| 1 | 1.58061940 | 1.24046873 | 1.274211 | 0.542941 |
| 2 | 1.64266094 | 1.24046873 | 1.324226 | 0.559475 |

## Baseline Aggregate

| policy | n | mse_mean | mse_std | selected_importance_mean | selected_importance_std |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned_selector_weighted_mse_alpha2 | 3 | 1.57323252 | 0.02413252 | 0.547769 | 0.008320 |
| random_k | 3 | 1.57845018 | 0.03707281 | 0.500644 | 0.002068 |
| teacher_importance_topk | 1 | 1.52461295 | 0.00000000 | 0.871909 | 0.000000 |
| uniform_k | 1 | 1.56802237 | 0.00000000 | 0.499789 | 0.000000 |

## Comparisons

- learned mean vs random mean MSE delta: `-0.00521766`
- learned mean vs uniform MSE delta: `0.00521015`
- learned mean vs teacher_importance_topk MSE delta: `0.04861957`
- learned selected importance vs random delta: `0.047126`
- learned selected importance vs uniform delta: `0.047981`

## Comparison To Step 14

```json
{
  "step14_train_samples": 500,
  "step14_test_samples": 64,
  "step14_baseline_learned_mse": 1.6451324323813121,
  "step14_baseline_random_mean_mse": 1.6592305037710402,
  "step14_baseline_uniform_mse": 1.677606850862503,
  "step15_learned_mse_mean_minus_step14_learned": -0.07189991242355775,
  "step15_random_mse_mean_minus_step14_random_mean": -0.08078032732009888,
  "step15_has_multiseed_learned_std": true
}
```

## Scientific Caveats

- learned weighted_mse_alpha2 mean MSE did not beat uniform_k MSE

## Sanity Gate

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

## Resource Summary

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

BAIR_1000_128_MULTI_SEED_VALIDATION_PASS = true


## Pytest

```text
........................................................................ [ 62%]
............................................                             [100%]
116 passed in 1.43s
```

BAIR_1000_128_MULTI_SEED_VALIDATION_PASS = true
