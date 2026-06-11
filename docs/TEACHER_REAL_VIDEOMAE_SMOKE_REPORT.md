# Teacher Real VideoMAE Smoke Report

Command: `python scripts/smoke_test_teacher_real_video_videomae.py`

- input token shard dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke`
- source encoder: `videomae`
- token shape: `[784, 768]`
- run dir: `/home/ubuntu22/tgpawb_world_model/runs/teacher_real_video_videomae_smoke_v1`
- checkpoint path: `/home/ubuntu22/tgpawb_world_model/runs/teacher_real_video_videomae_smoke_v1/checkpoints/teacher_world_model_step_000100.pt`
- num steps: `100`
- initial loss: `23.61896324157715`
- final loss: `0.21663761138916016`
- best loss: `0.10667528957128525`
- loss_decreased: `True`
- eval mse: `0.16617204993963242`
- GPU memory before: `{'free_gib': 14.512, 'total_gib': 15.461, 'used_gib': 0.948}`
- GPU memory after: `{'free_gib': 14.393, 'total_gib': 15.461, 'used_gib': 1.067}`
- RAM before: `{'total_gib': 30.403, 'available_gib': 25.764, 'used_gib': 4.156}`
- RAM after: `{'total_gib': 30.403, 'available_gib': 24.942, 'used_gib': 4.963}`
- elapsed time sec: `0.639`
- OOM: `False`
- cloud recommendation: `not required for Step 10A; local RTX 5080 smoke succeeded`

## Checks

```json
{
  "summary_exists": true,
  "metrics_exists": true,
  "checkpoint_exists": true,
  "num_steps_ok": true,
  "initial_loss_finite": true,
  "final_loss_finite": true,
  "best_loss_finite": true,
  "loss_decreased": true,
  "eval_mse_finite": true,
  "parameter_count_ok": true,
  "oom_ok": true
}
```

## Training Summary

```json
{
  "num_steps": 100,
  "initial_loss": 23.61896324157715,
  "final_loss": 0.21663761138916016,
  "best_loss": 0.10667528957128525,
  "loss_decreased": true,
  "loss_decreased_or_warn": true,
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_real_video_videomae_smoke_v1/checkpoints/teacher_world_model_step_000100.pt",
  "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_real_video_videomae_smoke_v1/metrics.jsonl",
  "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_real_video_videomae_smoke_v1/summary.json",
  "device": "cuda",
  "dataset_size": 8,
  "num_samples": 8,
  "num_shards": 4,
  "num_tokens": 784,
  "token_dim": 768,
  "past_token_shape": [
    784,
    768
  ],
  "future_token_shape": [
    784,
    768
  ],
  "source_encoder": "videomae",
  "source_split": "real_minimal",
  "token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke",
  "first_shard_path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke/tokens_shard_000000.pt",
  "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/teacher_real_video_videomae_smoke_v1"
}
```

## Eval Summary

```json
{
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_real_video_videomae_smoke_v1/checkpoints/teacher_world_model_step_000100.pt",
  "dataset_size": 8,
  "num_samples": 8,
  "num_tokens": 784,
  "token_dim": 768,
  "source_encoder": "videomae",
  "source_split": "real_minimal",
  "token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke",
  "num_batches": 4,
  "eval_mse": 0.16617204993963242,
  "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/teacher_real_video_videomae_smoke_v1"
}
```

TEACHER_REAL_VIDEOMAE_SMOKE_PASS = true
