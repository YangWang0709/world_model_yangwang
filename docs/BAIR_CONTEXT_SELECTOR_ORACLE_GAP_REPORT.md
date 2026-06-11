# BAIR Context Selector Oracle-Gap Report

Command: `python scripts/smoke_test_bair_context_selector_oracle_gap.py`

## Input Step17 Paths

- context tokens: `/home/ubuntu22/tgpawb_world_model/data/context_token_shards/bair_context_videomae_1000_128`
- context importance: `/home/ubuntu22/tgpawb_world_model/data/context_importance_shards/bair_context_teacher_1000_128`
- ContextTeacher checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/context_teacher_bair_1000_128_v1/checkpoints/context_teacher_step_001200.pt`
- Step17 baseline aggregate: `/home/ubuntu22/tgpawb_world_model/runs/context_bottleneck_baseline_bair_1000_128_v1/baseline_aggregate.json`

# BAIR Context Selector Oracle-Gap Summary

- stage: `context_selector_oracle_gap_bair_1000_128`
- context_topK: `32`
- current_tokens_dropped: `false`
- trained_current_importance: `false`
- trained_context_importance: `true`
- sanity gate pass: `true`
- cloud required: `false`

## Label Diagnostic

- raw importance mean/std/min/max: `-0.00003402` / `0.00036288` / `-0.00316894` / `0.00261378`
- norm importance mean/std/min/max: `0.52931654` / `0.18360086` / `0.00000000` / `1.00000000`
- positive ratio: `0.460260`
- top32 mass ratio: `0.06772450`
- oracle vs random importance gap: `0.33995044`
- temporal block distribution: `[5.118, 4.183, 5.25, 3.446, 4.337, 3.124, 3.844, 2.698]`

## Selector Phase A

| variant | loss_type | context_importance_mse | pearson_corr_mean | target_topk_overlap | selected_context_importance_mean | temporal_block_coverage |
| --- | --- | --- | --- | --- | --- | --- |
| weighted_mse_alpha2 | weighted_mse | 0.03312626 | 0.31632078 | 0.16040039 | 0.61387682 | 0.83593750 |
| weighted_mse_alpha5 | weighted_mse | 0.03548061 | 0.32017750 | 0.18481445 | 0.62228841 | 0.84472656 |
| topk_bce | topk_bce | 0.06196769 | 0.27631396 | 0.19726562 | 0.60230392 | 0.83007812 |
| pairwise_rank_w0p1 | pairwise_rank | 0.30211461 | 0.09694056 | 0.21386719 | 0.62246132 | 0.89160156 |
| hybrid_weighted_mse_rank_bce | hybrid_weighted_mse_rank_bce | 0.03674461 | 0.28087264 | 0.19824219 | 0.62486881 | 0.85839844 |
| weighted_mse_alpha2_no_current_condition | weighted_mse | 0.03052600 | 0.34605929 | 0.21728516 | 0.61326820 | 0.85351562 |
| weighted_mse_alpha2_no_temporal_pos | weighted_mse | 0.03226591 | 0.30769145 | 0.16821289 | 0.61585253 | 0.87011719 |
| temporal_block_balanced_topk | temporal_block_balanced_topk | 0.03696898 | 0.30564907 | 0.18774414 | 0.62596834 | 1.00000000 |

## Downstream Phase A

| variant | future_mse | beats_step17_learned | beats_random_context | oracle_gap |
| --- | --- | --- | --- | --- |
| weighted_mse_alpha2_no_current_condition | 1.39913664 | False | False | 0.18310616 |
| pairwise_rank_w0p1 | 1.38685546 | False | False | 0.17082498 |
| hybrid_weighted_mse_rank_bce | 1.39157588 | False | False | 0.17554540 |
| topk_bce | 1.40468086 | False | False | 0.18865038 |

## Selector Phase B

| variant | seeds | topk_overlap_mean | topk_overlap_std | selected_importance_mean | selected_importance_std |
| --- | --- | --- | --- | --- | --- |
| hybrid_weighted_mse_rank_bce | 0,1,2 | 0.20930989 | 0.01075390 | 0.62781262 | 0.00223778 |
| pairwise_rank_w0p1 | 0,1,2 | 0.21818034 | 0.00305149 | 0.62366945 | 0.00315211 |

## Downstream Phase B

| variant | seeds | future_mse_mean | future_mse_std | oracle_gap_mean |
| --- | --- | --- | --- | --- |
| hybrid_weighted_mse_rank_bce | 0,1,2 | 1.41154563 | 0.04177620 | 0.19551511 |
| pairwise_rank_w0p1 | 0,1,2 | 1.40233290 | 0.03055387 | 0.18630241 |

## Oracle Gap

- Step17 learned MSE: `1.37120354`
- Step17 random MSE: `1.36507559`
- Step17 current_only MSE: `1.39464128`
- Step17 teacher oracle MSE: `1.21603048`
- Step18 best variant: `pairwise_rank_w0p1`
- Step18 best MSE mean: `1.40233290`
- oracle gap after Step18: `0.18630242`
- oracle gap reduction vs Step17: `-0.03112936`

## Diagnosis

- label_sparse_or_noisy: `true`
- topk_overlap_bottleneck: `true`
- current_conditioning_helpful: `false`
- temporal_position_helpful: `false`
- temporal_block_balancing_helpful: `false`
- downstream_utilization_still_bottleneck: `true`

## Next Recommendation

Labels look diffuse or noisy on BAIR; prefer a longer-context dataset such as BridgeData or DROID before claiming long-memory value.

## Sanity Gate

```json
{
  "checks": {
    "step17_inputs_exist": true,
    "label_diagnostic_generated": true,
    "phase_a_selector_variants_recorded": true,
    "phase_b_two_variants_three_seeds": true,
    "metrics_finite": true,
    "current_tokens_dropped_false": true,
    "trained_current_importance_false": true,
    "trained_context_importance_true": true,
    "pytest_passed": true,
    "env_isaaclab_unpolluted": true,
    "oom_false": true
  },
  "pass": true
}
```

## Resource Summary

```json
{
  "before": {
    "disk_total_gib": 483.587,
    "disk_used_gib": 303.297,
    "disk_free_gib": 160.395,
    "ram_total_gib": 30.403,
    "ram_available_gib": 25.507,
    "ram_used_gib": 4.286,
    "torch_cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.461,
    "gpu_mem_used_gib": 0.962,
    "gpu_mem_free_gib": 14.499
  },
  "after": {
    "disk_total_gib": 483.587,
    "disk_used_gib": 303.822,
    "disk_free_gib": 159.87,
    "ram_total_gib": 30.403,
    "ram_available_gib": 18.918,
    "ram_used_gib": 10.858,
    "torch_cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.461,
    "gpu_mem_used_gib": 1.19,
    "gpu_mem_free_gib": 14.27
  },
  "elapsed_time_sec": 60.992,
  "max_ram_used_gib": 10.858,
  "max_gpu_mem_used_gib": 1.19,
  "gpu_mem_fraction": 0.07696785460190156,
  "oom": false,
  "cloud_recommendation": "not required for Step18 local oracle-gap diagnostic"
}
```

BAIR_CONTEXT_SELECTOR_ORACLE_GAP_PASS = true


## Pytest

```text
........................................................................ [ 48%]
........................................................................ [ 97%]
....                                                                     [100%]
=============================== warnings summary ===============================
../miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/torch/nn/modules/transformer.py:382
  /home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/torch/nn/modules/transformer.py:382: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
148 passed, 1 warning in 1.45s
```

BAIR_CONTEXT_SELECTOR_ORACLE_GAP_PASS = true
