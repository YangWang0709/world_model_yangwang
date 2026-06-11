# STEP12 BAIR Baseline Comparison Report

## 1. Goal

Step 12 compares token selection policies on BAIR Robot Pushing small represented as VideoMAE token shards.

## 2. Cloud Server Decision

No cloud server is required for this stage because the run reads existing token and importance shards and does not run the VideoMAE encoder.

- OOM: `False`
- max RAM used GiB: `8.207664489746094`
- max GPU memory used GiB: `1.109375`
- cloud recommendation: `not required for Step 12 smoke`

## 3. Why Baselines Are Needed

Step 11F proved the learned-selector compressed Student pipeline runs end to end. Step 12 checks whether the learned selector compares favorably to Random-K, Uniform-K, and an oracle-like Teacher-Importance TopK baseline at the same token budget.

## 4. Baselines

- Full-token Teacher reference
- Random-K Student
- Uniform-K Student
- Teacher-Importance TopK Student
- Learned Selector Student

## 5. Training Protocol

- topK: `16`
- token retention ratio: `16 / 392 = 0.04081632653061224`
- max_steps per baseline: `500`
- batch_size: `4`
- all Student baselines train independent `TokenCompressor + StudentWorldModel` modules
- Teacher and learned selector remain frozen

## 6. Metrics

- student_future_mse
- teacher_mse
- student_teacher_ratio
- token_retention_ratio
- selector_target_topK_overlap
- selected_teacher_importance_mean
- selected_vs_random_importance_gap

## 7. Results Table

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


## 8. Sanity Gate

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

## 9. Pytest Result

`88 passed in 1.44s`

## 10. Git Commit

Branch: `feature/tgpawb-step3-token-extraction`. The final commit hash and push status are recorded after commit/push in `STEP12_LOCAL_SUMMARY.md` and the final response. No PR is created.

## 11. What Was Not Done

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

## 12. Next Step Recommendation

Step 13 should choose the next direction from the baseline result: expand BAIR if learned selection is stable, improve selector supervision if it is not, or add task/action conditioning if the paper direction needs stronger task grounding.
