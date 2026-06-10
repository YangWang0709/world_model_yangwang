# STEP8 Baseline Comparison Report

## 1. Goal

Compare compressed Student world-model prediction on `structured_toy` under different token selection policies at the same token budget.

## 2. Why Baselines Are Needed

Step 7 showed the learned selector pipeline can train. Step 8 checks whether that selector is better than Random-K / Uniform-K and close to an Oracle-Key upper-bound under the same Student training protocol.

## 3. Baselines

- Full-token Teacher reference
- Random-K Student
- Uniform-K Student
- Oracle-Key Student
- Learned Selector Student

## 4. Training Protocol

All Student baselines use `topK=4`, the same `TokenCompressor`, the same `StudentWorldModel`, the same structured toy shards, and the same `max_steps=300` training budget. The teacher and learned selector are frozen references.

## 5. Metrics

- future_mse
- teacher_mse
- student_teacher_ratio
- token_retention_ratio
- selected_top1_hit_rate
- selected_topk_hit_rate
- selected_key_coverage

## 6. Results Table

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


## 7. Sanity Gate

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

Smoke gate pass: `true`

## 8. Pytest Result

`/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python -m pytest tests -q` completed with `34 passed in 0.98s`.

The regression smoke sequence also passed after Step 8 changes: minimal pipeline, token extraction, teacher training, predictive importance, structured importance signal quality, Student selector training, Student world-model training, and baseline comparison.

## 9. Git Commit

Branch: `feature/tgpawb-step3-token-extraction`. Commit hash and push status are recorded after commit/push. No PR is created in this stage.

## 10. What Was Not Done

- no real dataset download
- no real V-JEPA / VideoMAE encoder
- no large model download
- no VLM grounding
- no RL / policy optimization
- no action-conditioned world model
- no generated `.pt` committed
- no checkpoint committed
- no password or token saved
- no PR created

## 11. Next Step Recommendation

Step 9 should prepare a minimal real-video subset and loader before introducing frozen VideoMAE/V-JEPA token extraction.

The structured toy result is a sanity check rather than a paper-level result. Learned selection clearly beats Random-K on future MSE and key-token coverage, and matches Oracle-Key on key coverage. Uniform-K showed lower future MSE in this tiny train/eval-on-same-structured-toy run despite low key coverage, so that artifact should be rechecked with a held-out or less synthetic set before drawing broader conclusions.

## Artifact Paths

- run dir: `/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_structured_toy_v1`
- baseline summary json: `/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_structured_toy_v1/baseline_summary.json`
- baseline summary csv: `/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_structured_toy_v1/baseline_summary.csv`
- baseline summary md: `/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_structured_toy_v1/baseline_summary.md`
