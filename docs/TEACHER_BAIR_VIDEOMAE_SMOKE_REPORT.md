# Teacher BAIR VideoMAE Smoke Report

- command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_teacher_bair_videomae.py`
- train token shard dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train`
- test token shard dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test`
- train samples: `100`
- test samples: `16`
- token shape: `[100, 392, 768]`
- run dir: `/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1`
- checkpoint path: `/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1/checkpoints/teacher_world_model_step_000300.pt`
- num steps: `300`
- initial loss: `13.67411994934082`
- final loss: `0.86194908618927`
- best loss: `0.6186790466308594`
- loss_decreased: `True`
- loss_warning: `None`
- eval_mse: `1.686881422996521`
- elapsed_time_sec: `1.2`
- oom: `False`
- cloud_recommendation: `not_needed_for_step11c`
- TEACHER_BAIR_VIDEOMAE_SMOKE_PASS: `true`

## Resource Before

```json
{
  "disk_free_gib": 188.79,
  "gpu": {
    "nvidia_smi": [
      {
        "memory_free_mib": 15042,
        "memory_total_mib": 16303,
        "memory_used_mib": 791,
        "name": "NVIDIA GeForce RTX 5080"
      }
    ],
    "torch_cuda_allocated_mib": 0.0,
    "torch_cuda_available": true,
    "torch_cuda_reserved_mib": 0.0
  },
  "ram_available_gib": 24.698,
  "ram_total_gib": 30.403,
  "ram_used_gib": 5.705
}
```

## Resource After

```json
{
  "disk_free_gib": 188.782,
  "gpu": {
    "nvidia_smi": [
      {
        "memory_free_mib": 14654,
        "memory_total_mib": 16303,
        "memory_used_mib": 1178,
        "name": "NVIDIA GeForce RTX 5080"
      }
    ],
    "torch_cuda_allocated_mib": 25.282,
    "torch_cuda_available": true,
    "torch_cuda_reserved_mib": 62.0
  },
  "ram_available_gib": 23.865,
  "ram_total_gib": 30.403,
  "ram_used_gib": 6.538
}
```

## Training Summary

```json
{
  "best_loss": 0.6186790466308594,
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1/checkpoints/teacher_world_model_step_000300.pt",
  "dataset": "bair_robot_pushing_small",
  "dataset_size": 100,
  "device": "cuda",
  "final_loss": 0.86194908618927,
  "first_shard_path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000000.pt",
  "future_token_shape": [
    392,
    768
  ],
  "initial_loss": 13.67411994934082,
  "loss_decreased": true,
  "loss_decreased_or_warn": true,
  "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1/metrics.jsonl",
  "num_samples": 100,
  "num_shards": 25,
  "num_steps": 300,
  "num_tokens": 392,
  "past_token_shape": [
    392,
    768
  ],
  "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1",
  "source_encoder": "videomae",
  "source_split": "train",
  "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1/summary.json",
  "test_dataset_summary": {
    "dataset": "bair_robot_pushing_small",
    "first_shard_path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test/tokens_shard_000000.pt",
    "future_token_shape": [
      392,
      768
    ],
    "num_samples": 16,
    "num_shards": 4,
    "num_tokens": 392,
    "past_token_shape": [
      392,
      768
    ],
    "source_encoder": "videomae",
    "source_split": "test",
    "token_dim": 768,
    "token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test"
  },
  "test_num_samples": 16,
  "test_token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test",
  "token_dim": 768,
  "token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train",
  "train_num_samples": 100,
  "train_token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train"
}
```

## Eval Summary

```json
{
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1/checkpoints/teacher_world_model_step_000300.pt",
  "checkpoint_step": 300,
  "dataset": "bair_robot_pushing_small",
  "dataset_size": 16,
  "device": "cuda",
  "eval_mse": 1.686881422996521,
  "num_batches": 4,
  "num_samples": 16,
  "num_tokens": 392,
  "pred_max": 87.75140380859375,
  "pred_mean": 0.2887116074562073,
  "pred_min": -6.821407794952393,
  "pred_std": 3.3462746143341064,
  "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1",
  "source_encoder": "videomae",
  "source_split": "test",
  "split": "test",
  "target_max": 96.4380874633789,
  "target_mean": 0.3002580404281616,
  "target_min": -9.533923149108887,
  "target_std": 3.6612801551818848,
  "token_dim": 768,
  "token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test"
}
```

## Checkpoint Summary

```json
{
  "checkpoint_keys": [
    "metrics_summary",
    "model_config",
    "model_state_dict",
    "optimizer_state_dict",
    "step"
  ],
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1/checkpoints/teacher_world_model_step_000300.pt",
  "metrics_summary": {
    "best_loss": 0.6186790466308594,
    "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1/checkpoints/teacher_world_model_step_000300.pt",
    "dataset": "bair_robot_pushing_small",
    "dataset_size": 100,
    "device": "cuda",
    "final_loss": 0.86194908618927,
    "first_shard_path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000000.pt",
    "future_token_shape": [
      392,
      768
    ],
    "initial_loss": 13.67411994934082,
    "loss_decreased": true,
    "loss_decreased_or_warn": true,
    "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1/metrics.jsonl",
    "num_samples": 100,
    "num_shards": 25,
    "num_steps": 300,
    "num_tokens": 392,
    "past_token_shape": [
      392,
      768
    ],
    "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1",
    "source_encoder": "videomae",
    "source_split": "train",
    "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1/summary.json",
    "test_dataset_summary": {
      "dataset": "bair_robot_pushing_small",
      "first_shard_path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test/tokens_shard_000000.pt",
      "future_token_shape": [
        392,
        768
      ],
      "num_samples": 16,
      "num_shards": 4,
      "num_tokens": 392,
      "past_token_shape": [
        392,
        768
      ],
      "source_encoder": "videomae",
      "source_split": "test",
      "token_dim": 768,
      "token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test"
    },
    "test_num_samples": 16,
    "test_token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test",
    "token_dim": 768,
    "token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train",
    "train_num_samples": 100,
    "train_token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train"
  },
  "model_config": {
    "dropout": 0.0,
    "hidden_dim": 512,
    "num_layers": 2,
    "output_dim": 768,
    "pool": "mean",
    "token_dim": 768
  },
  "parameter_count": 789248,
  "state_dict_key_count": 6,
  "step": 300
}
```

TEACHER_BAIR_VIDEOMAE_SMOKE_PASS = true
