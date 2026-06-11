# BAIR Baseline Comparison Smoke Report

Command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_bair_baseline_comparison.py`

- run dir: `/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1`
- policies: `random_k, uniform_k, teacher_importance_topk, learned_selector`
- summary_json: `/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/baseline_summary.json`
- summary_csv: `/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/baseline_summary.csv`
- summary_md: `/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/baseline_summary.md`

## Baseline Summary Table

| policy | seed | final_loss | student_future_mse | teacher_mse | student_teacher_ratio | token_retention_ratio | selector_target_topk_overlap | selected_teacher_importance_mean | selected_vs_random_importance_gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned_selector | 0 | 1.38129079 | 1.84130295 | 1.68688138 | 1.091543 | 0.040816 | 0.164062 | 0.592747 | 0.134254 |
| random_k | 0 | 1.40566635 | 1.88762669 | 1.68688138 | 1.119004 | 0.040816 | 0.027344 | 0.458493 | 0.000000 |
| random_k | 1 | 1.74472654 | 1.86600073 | 1.68688138 | 1.106184 | 0.040816 | 0.046875 | 0.477553 | 0.000000 |
| random_k | 2 | 1.29204988 | 1.88074883 | 1.68688138 | 1.114927 | 0.040816 | 0.035156 | 0.476636 | 0.000000 |
| teacher_importance_topk | 0 | 1.37759626 | 1.76087014 | 1.68688138 | 1.043861 | 0.040816 | 0.996094 | 0.835729 | 0.377236 |
| uniform_k | 0 | 1.36509216 | 1.83501625 | 1.68688138 | 1.087816 | 0.040816 | 0.054688 | 0.482644 | 0.024152 |

## Aggregate

```json
{
  "random_k_student_future_mse": {
    "mean": 1.8781254159079657,
    "std": 0.009021537614255172
  },
  "random_k_selected_teacher_importance_mean": {
    "mean": 0.4708937505880992,
    "std": 0.008776986411320188
  },
  "random_k_selected_key_coverage": {
    "mean": null,
    "std": null
  },
  "learned_vs_random": {
    "student_future_mse_delta": -0.03682246473100448,
    "future_mse_lower_than_random_mean": true,
    "selected_teacher_importance_delta": 0.12185292442639667,
    "selected_teacher_importance_higher_than_random_mean": true,
    "selected_key_coverage_delta": null,
    "coverage_higher_than_random_mean": false
  },
  "learned_vs_oracle": {
    "student_future_mse_gap": 0.08043281237284328,
    "selected_teacher_importance_gap": -0.24298197031021118,
    "selected_key_coverage_gap": null
  },
  "sanity_gate": {
    "checks": {
      "learned_selector_present": true,
      "random_k_present": true,
      "oracle_present": true,
      "learned_ratio_finite": true,
      "learned_selected_importance_gt_random_mean": true,
      "learned_selected_vs_random_gap_positive": true,
      "oracle_importance_ge_learned": true
    },
    "pass": true
  }
}
```


## Learned vs Random

```json
{
  "student_future_mse_delta": -0.03682246473100448,
  "future_mse_lower_than_random_mean": true,
  "selected_teacher_importance_delta": 0.12185292442639667,
  "selected_teacher_importance_higher_than_random_mean": true,
  "selected_key_coverage_delta": null,
  "coverage_higher_than_random_mean": false
}
```

## Learned vs Teacher-Importance TopK

```json
{
  "student_future_mse_gap": 0.08043281237284328,
  "selected_teacher_importance_gap": -0.24298197031021118,
  "selected_key_coverage_gap": null
}
```

## Sanity Gate

```json
{
  "checks": {
    "learned_selector_present": true,
    "random_k_present": true,
    "oracle_present": true,
    "learned_ratio_finite": true,
    "learned_selected_importance_gt_random_mean": true,
    "learned_selected_vs_random_gap_positive": true,
    "oracle_importance_ge_learned": true
  },
  "pass": true
}
```

## Resource Usage

```json
{
  "before": {
    "ram_used_gib": 6.628353118896484,
    "ram_total_gib": 30.40261459350586,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.9208984375,
    "gpu_mem_used_gib": 0.6728515625
  },
  "after": {
    "ram_used_gib": 8.207664489746094,
    "ram_total_gib": 30.40261459350586,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.9208984375,
    "gpu_mem_used_gib": 1.109375
  },
  "elapsed_time_sec": 9.137208700180054,
  "max_ram_used_gib": 8.207664489746094,
  "max_gpu_mem_used_gib": 1.109375,
  "oom": false,
  "gpu_mem_fraction": 0.06968042691529167,
  "cloud_recommendation": "not required for Step 12 smoke"
}
```

## Checks

```json
{
  "baseline_summary_json_exists": true,
  "baseline_summary_csv_exists": true,
  "baseline_summary_md_exists": true,
  "required_policies_present": true,
  "learned_selector_ratio_finite": true,
  "learned_selector_importance_finite": true,
  "teacher_importance_topk_importance_finite": true,
  "sanity_gate_pass": true,
  "ram_below_warning": true
}
```

## Pytest Result

`88 passed in 1.44s`

BAIR_BASELINE_COMPARISON_SMOKE_PASS = true
