# STEP17 BAIR Context Bottleneck World Model

## 1. Goal

Validate the shift from state/current bottlenecking to historical context bottlenecking on BAIR 1000/128.

## 2. Motivation

The current observation is kept complete and only summarized. Predictive importance selection is applied to historical context tokens only.

## 3. Key Design Decision

- current_tokens are not topK-dropped
- current importance is not trained in this step
- only context importance is trained
- UnifiedPredictiveImportanceSelector is used in context mode
- state_legacy mode is retained only for compatibility

## 4. Cloud Server Decision

Cloud server required: `false`. Local execution is sufficient unless OOM persists after reducing batch size or token_chunk_size.

## 5. Dataset

- train/test: `1000` / `128`
- context/current/future: `8` / `4` / `4` frames
- action/endeffector are saved as metadata and are not model inputs

## 6. Token Extraction

- context_topK: `32`
- context retention ratio: `0.04081632653061224`
- VideoMAE local checkpoint only; no fallback

## 7. Context Teacher

- eval MSE: `1.37276897`

## 8. Context Importance

I_i = L(Teacher(context_without_i, current_full), future) - L(Teacher(context_full, current_full), future)

## 9. Unified Predictive Importance Selector

- importance MSE: `0.03289004`
- pearson corr: `0.30424362`
- target topK overlap: `0.15966797`

## 10. Context Bottleneck World Model

- learned_context MSE: `1.37120354`
- current_only MSE: `1.39464128`
- context gain: `0.02343774`

## 11. Baselines

# BAIR Context Bottleneck Validation Summary

- stage: `context_bottleneck_bair_1000_128`
- train/test samples: `1000` / `128`
- context/current/future frames: `8` / `4` / `4`
- context_topK: `32`
- context retention ratio: `0.040816326531`
- current_tokens_dropped: `false`
- trained_current_importance: `false`
- trained_context_importance: `true`
- sanity gate pass: `true`
- cloud required: `false`

## Baselines

| policy | seeds | n | mse_mean | mse_std | selected_context_importance | topK_overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| teacher_context_importance_topK | 0 | 1 | 1.21603048 | 0.00000000 | 0.866173 | 1.000000 |
| random_context_topK | 0,1,2 | 3 | 1.36507559 | 0.03989352 | 0.517025 | 0.040690 |
| learned_context_selector_topK | 0,1,2 | 3 | 1.37120354 | 0.02141377 | 0.609647 | 0.159912 |
| hybrid_context_learned_uniform | 0,1,2 | 3 | 1.37246382 | 0.01629830 | 0.563953 | 0.114746 |
| full_context_teacher_reference | 0 | 1 | 1.37276900 | 0.00000000 | 0.000000 | 0.000000 |
| current_only | 0 | 1 | 1.39464128 | 0.00000000 | 0.479678 | 0.000000 |
| uniform_context_topK | 0 | 1 | 1.46179116 | 0.00000000 | 0.513461 | 0.040039 |

## Comparison

- context teacher MSE: `1.37276897`
- current_only MSE: `1.39464128`
- learned_context MSE: `1.37120354`
- random_context MSE: `1.36507559`
- uniform_context MSE: `1.46179116`
- teacher_context_importance_topK MSE: `1.21603048`
- hybrid_context MSE: `1.37246382`
- context gain over current_only: `0.02343774`
- oracle gap: `0.15517306`

## Interpretation

- learned beats current_only: `true`
- learned beats random: `false`
- learned beats uniform: `true`
- hybrid beats learned: `false`
- BAIR limitation: BAIR context window validates the context bottleneck pipeline, but may not be sufficient to prove long-context memory value.

## Scientific Caveats

- learned context did not clearly beat random context topK

## Sanity Gate

```json
{
  "checks": {
    "context_windows_exported": true,
    "context_tokens_extracted": true,
    "context_teacher_trained": true,
    "context_importance_generated": true,
    "unified_selector_trained_context_only": true,
    "world_model_trained": true,
    "baseline_comparison_complete": true,
    "current_tokens_dropped_false": true,
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
    "disk_used_gib": 287.678,
    "disk_free_gib": 176.014,
    "ram_total_gib": 30.403,
    "ram_available_gib": 25.269,
    "ram_used_gib": 4.576,
    "torch_cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.461,
    "gpu_mem_used_gib": 0.964,
    "gpu_mem_free_gib": 14.497
  },
  "after": {
    "disk_total_gib": 483.587,
    "disk_used_gib": 303.286,
    "disk_free_gib": 160.406,
    "ram_total_gib": 30.403,
    "ram_available_gib": 12.197,
    "ram_used_gib": 17.553,
    "torch_cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.461,
    "gpu_mem_used_gib": 1.799,
    "gpu_mem_free_gib": 13.661
  },
  "elapsed_time_sec": 183.093,
  "max_ram_used_gib": 17.553,
  "max_gpu_mem_used_gib": 1.799,
  "gpu_mem_fraction": 0.11635728607463941,
  "oom": false,
  "cloud_recommendation": "not required for Step 17 local context pipeline validation"
}
```

BAIR_CONTEXT_BOTTLENECK_VALIDATION_PASS = true


## 12. BAIR Context Limitation

BAIR context window validates the context bottleneck pipeline, but may not be sufficient to prove long-context memory value.

## 13. Sanity Gate

```json
{
  "checks": {
    "context_windows_exported": true,
    "context_tokens_extracted": true,
    "context_teacher_trained": true,
    "context_importance_generated": true,
    "unified_selector_trained_context_only": true,
    "world_model_trained": true,
    "baseline_comparison_complete": true,
    "current_tokens_dropped_false": true,
    "pytest_passed": true,
    "env_isaaclab_unpolluted": true,
    "oom_false": true
  },
  "pass": true
}
```

## 14. Pytest

- command: `/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python -m pytest tests -q`
- passed: `true`

## 15. What Was Not Done

- no new model download
- no new dataset download
- no BAIR re-download
- no VideoMAE training
- no current importance training
- no current token topK dropping
- no Step11-16 overwrite
- no VLM grounding
- no RL
- no action-conditioned world model
- action/endeffector saved but not used as input
- no model weight committed
- no BAIR/context data committed
- no token/importance shard committed
- no checkpoint committed
- no password/token saved
- no PR created
