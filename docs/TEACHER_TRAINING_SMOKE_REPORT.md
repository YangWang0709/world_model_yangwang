# Teacher Training Smoke Report

Command: `python scripts/smoke_test_teacher_training.py`

- token shard path: `/home/ubuntu22/tgpawb_world_model/data/token_shards/dummy_toy`
- run dir: `/home/ubuntu22/tgpawb_world_model/runs/teacher_dummy_tiny_v1`
- checkpoint path: `/home/ubuntu22/tgpawb_world_model/runs/teacher_dummy_tiny_v1/checkpoints/teacher_world_model_step_000030.pt`
- metrics path: `/home/ubuntu22/tgpawb_world_model/runs/teacher_dummy_tiny_v1/metrics.jsonl`
- num steps: `30`
- initial loss: `0.03960206359624863`
- final loss: `9.57007723627612e-05`
- best loss: `9.57007723627612e-05`
- loss_decreased: `True`
- device: `cuda`
- eval mse: `8.54052177601261e-05`

## Checkpoint Inspection Summary

```json
{
  "step": 30,
  "parameter_count": 789248,
  "state_dict_key_count": 6,
  "model_config": {
    "token_dim": 768,
    "hidden_dim": 512,
    "output_dim": 768,
    "num_layers": 2,
    "dropout": 0.0,
    "pool": "mean"
  }
}
```

TEACHER_TRAINING_SMOKE_PASS = true
