# Baseline Comparison Smoke Report

Command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_baseline_comparison.py`

- run dir: `/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_structured_toy_v1`
- policies: `['random_k', 'uniform_k', 'oracle_key', 'learned_selector']`
- summary json: `/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_structured_toy_v1/baseline_summary.json`
- summary csv: `/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_structured_toy_v1/baseline_summary.csv`
- summary md: `/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_structured_toy_v1/baseline_summary.md`

## Baseline Summary Table

| policy | seed | final_loss | student_future_mse | teacher_future_mse | student_teacher_ratio | token_retention_ratio | selected_top1_hit_rate | selected_topk_hit_rate | selected_key_coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned_selector | 0 | 0.00271330 | 0.00262337 | 0.00269326 | 0.974047 | 0.020408 | 1.000000 | 1.000000 | 1.000000 |
| oracle_key | 0 | 0.00281371 | 0.00248906 | 0.00269326 | 0.924181 | 0.020408 | 1.000000 | 1.000000 | 1.000000 |
| random_k | 0 | 0.27551714 | 0.28362492 | 0.00269326 | 105.308981 | 0.020408 | 0.000000 | 0.015625 | 0.015625 |
| random_k | 1 | 0.51335448 | 0.19128878 | 0.00269326 | 71.024880 | 0.020408 | 0.000000 | 0.007812 | 0.007812 |
| random_k | 2 | 0.23446839 | 0.24499598 | 0.00269326 | 90.966182 | 0.020408 | 0.031250 | 0.015625 | 0.015625 |
| uniform_k | 0 | 0.00176178 | 0.00134206 | 0.00269326 | 0.498301 | 0.020408 | 0.000000 | 0.023438 | 0.023438 |

## Aggregate

```json
{
  "random_k_student_future_mse": {
    "mean": 0.2399698926342858,
    "std": 0.037863238082482875
  },
  "random_k_selected_key_coverage": {
    "mean": 0.013020833333333334,
    "std": 0.003682847818679935
  },
  "learned_vs_random": {
    "student_future_mse_delta": -0.23734652632588726,
    "selected_key_coverage_delta": 0.9869791666666666,
    "future_mse_lower_than_random_mean": true,
    "coverage_higher_than_random_mean": true
  },
  "learned_vs_oracle": {
    "student_future_mse_gap": 0.00013430137187242508,
    "selected_key_coverage_gap": 0.0
  },
  "sanity_gate": {
    "checks": {
      "learned_selector_present": true,
      "random_k_present": true,
      "oracle_key_present": true,
      "learned_topk_hit_rate": true,
      "learned_key_coverage": true,
      "learned_future_mse_le_random_mean": true,
      "learned_ratio_finite": true
    },
    "pass": true
  }
}
```


## Learned Selector vs Random-K

```json
{
  "student_future_mse_delta": -0.23734652632588726,
  "selected_key_coverage_delta": 0.9869791666666666,
  "future_mse_lower_than_random_mean": true,
  "coverage_higher_than_random_mean": true
}
```

## Learned Selector vs Oracle-Key

```json
{
  "student_future_mse_gap": 0.00013430137187242508,
  "selected_key_coverage_gap": 0.0
}
```

## Sanity Gate

```json
{
  "checks": {
    "learned_selector_present": true,
    "random_k_present": true,
    "oracle_key_present": true,
    "learned_topk_hit_rate": true,
    "learned_key_coverage": true,
    "learned_future_mse_le_random_mean": true,
    "learned_ratio_finite": true
  },
  "pass": true
}
```

BASELINE_COMPARISON_SMOKE_PASS = true
