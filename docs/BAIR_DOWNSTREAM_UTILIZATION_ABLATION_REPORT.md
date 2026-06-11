# BAIR Downstream Utilization Ablation Report

Command: `python scripts/smoke_test_bair_downstream_utilization_ablation.py`

## Run Dirs

- Step16 run: `/home/ubuntu22/tgpawb_world_model/runs/downstream_utilization_bair_1000_128_v1`
- Phase A: `/home/ubuntu22/tgpawb_world_model/runs/downstream_utilization_bair_1000_128_v1/phase_a`
- Phase B: `/home/ubuntu22/tgpawb_world_model/runs/downstream_utilization_bair_1000_128_v1/phase_b`

## Fixed Step15 Inputs

- train_token_shard_dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_1000_128/train`
- test_token_shard_dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_1000_128/test`
- train_importance_shard_dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_1000_128/train`
- test_importance_shard_dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_1000_128/test`
- teacher_checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_1000_128_v1/checkpoints/teacher_world_model_step_001200.pt`
- selector_root: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_1000_128_weighted_mse_multiseed_v1`
- baseline_aggregate: `/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_1000_128_multiseed_v1/baseline_multiseed_aggregate.json`

## Unified Summary

# BAIR Downstream Utilization Ablation Summary

- stage: `downstream_utilization_bair_1000_128`
- train/test samples: `1000` / `128`
- topK: `16`
- token retention ratio: `0.040816326531`
- phase A best: `selector_score_weighted_pool`
- phase B best: `mean_pool_selected`
- phase B best MSE mean/std: `1.39506111` / `0.00904936`
- sanity gate pass: `true`
- cloud required: `false`

## Phase A

| variant | mse | ratio | selected_importance | topK_overlap |
| --- | ---: | ---: | ---: | ---: |
| selector_score_weighted_pool | 1.42233196 | 1.146608 | 0.540891 | 0.110352 |
| mean_pool_selected | 1.42345227 | 1.147512 | 0.540891 | 0.110352 |
| attention_pool_selected | 1.56332328 | 1.260268 | 0.540891 | 0.110352 |
| current_perceiver_like | 1.59231013 | 1.283636 | 0.540891 | 0.110352 |
| hybrid_learned8_uniform8_cross_attention | 1.61138300 | 1.299011 | 0.519385 | 0.089844 |
| hybrid_learned8_uniform8_perceiver | 1.61423208 | 1.301308 | 0.519385 | 0.089844 |
| transformer_encoder_selected | 1.62044428 | 1.306316 | 0.540891 | 0.110352 |
| cross_attention_latent_bottleneck | 1.62373509 | 1.308969 | 0.540891 | 0.110352 |

## Phase B

| variant | seeds | mse_mean | mse_std | ratio_mean | selected_importance_mean | topK_overlap_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean_pool_selected | 0,1,2 | 1.39506111 | 0.00904936 | 1.124624 | 0.547769 | 0.115397 |
| selector_score_weighted_pool | 0,1,2 | 1.39673269 | 0.00993852 | 1.125972 | 0.547769 | 0.115397 |

## Step15 Comparison

- best vs Step15 learned MSE delta: `-0.17817141`
- best vs Step15 random mean MSE delta: `-0.18338907`
- best vs Step15 uniform MSE delta: `-0.17296126`
- best vs teacher_importance_topk MSE delta: `-0.12955184`
- best beats Step15 learned: `true`
- best beats Step15 random mean: `true`
- best beats Step15 uniform: `true`

## Scientific Caveats

- best Step16 selected importance is below Step15 learned selected importance mean
- no hybrid variant reached Phase B top-2

## Sanity Gate

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

## Resource Summary

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

BAIR_DOWNSTREAM_UTILIZATION_ABLATION_PASS = true


## Interpretation

- best variant: `mean_pool_selected`
- best Step16 MSE: `1.39506111`
- current compressor bottleneck signal: `true` for beating Step15 learned; `true` for beating uniform
- hybrid learned+uniform reached Phase B: `false`
- cross-attention reached Phase B: `false`
- Step17 recommendation: use the best Step16 utilizer if it beats uniform; otherwise prioritize action-conditioned world model.

## Pytest

```text
........................................................................ [ 55%]
..........................................................               [100%]
=============================== warnings summary ===============================
../miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/torch/nn/modules/transformer.py:382
  /home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/torch/nn/modules/transformer.py:382: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
130 passed, 1 warning in 1.44s
```

BAIR_DOWNSTREAM_UTILIZATION_ABLATION_PASS = true
