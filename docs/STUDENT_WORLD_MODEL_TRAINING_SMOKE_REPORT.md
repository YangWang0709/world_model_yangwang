# Student World Model Training Smoke Report

Command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_student_world_model_training.py`

- token shard input dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/structured_toy`
- importance shard input dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/structured_toy_teacher`
- selector checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_structured_toy_v1/checkpoints/student_selector_step_000200.pt`
- teacher checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/teacher_structured_toy_tiny_v1/checkpoints/teacher_world_model_step_000100.pt`
- run dir: `/home/ubuntu22/tgpawb_world_model/runs/student_world_model_structured_toy_v1`
- checkpoint path: `/home/ubuntu22/tgpawb_world_model/runs/student_world_model_structured_toy_v1/checkpoints/student_world_model_step_000300.pt`
- num steps: `300`
- initial loss: `0.4130794405937195`
- final loss: `0.002919309539720416`
- best loss: `0.002136853989213705`
- loss_decreased: `True`
- student future MSE: `0.0023518727781871953`
- teacher future MSE: `0.002693264357124766`
- student_teacher_gap: `-0.00034139157893757054`
- student_teacher_ratio: `0.8732424546315134`
- selected top1 hit rate: `1.0`
- selected topk hit rate: `1.0`
- selected key coverage: `1.0`
- token retention ratio: `0.02040816326530612`

## Training Summary

```json
{
  "num_steps": 300,
  "initial_loss": 0.4130794405937195,
  "final_loss": 0.002919309539720416,
  "best_loss": 0.002136853989213705,
  "loss_decreased": true,
  "student_future_mse": 0.0023518727781871953,
  "final_selected_top1_hit_rate": 1.0,
  "final_selected_topk_hit_rate": 1.0,
  "final_selected_key_coverage": 1.0,
  "final_selected_key_fraction": 1.0,
  "token_retention_ratio": 0.02040816326530612,
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_structured_toy_v1/checkpoints/student_world_model_step_000300.pt",
  "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_structured_toy_v1/metrics.jsonl",
  "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_structured_toy_v1/summary.json",
  "device": "cuda",
  "dataset_size": 32,
  "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_structured_toy_v1",
  "selector_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_structured_toy_v1/checkpoints/student_selector_step_000200.pt",
  "selector_frozen": true,
  "topk": 4,
  "compressed_tokens": 8,
  "student_world_model_eval": {
    "student_future_mse": 0.0023518727781871953,
    "prediction_mean": 0.00019946768588852137,
    "prediction_std": 0.5935211181640625,
    "target_mean": -0.00020117669191677123,
    "target_std": 0.591864287853241,
    "key_score_mean": 0.6109598278999329,
    "non_key_score_mean": 0.010969343595206738,
    "key_vs_non_key_gap": 0.5999904843047261,
    "top1_hit_rate": 1.0,
    "topk_hit_rate": 1.0,
    "selected_key_coverage": 1.0,
    "selected_key_fraction": 1.0
  },
  "smoke_checks": {
    "summary_exists": true,
    "metrics_exists": true,
    "checkpoint_exists": true,
    "num_steps_ok": true,
    "initial_loss_finite": true,
    "final_loss_finite": true,
    "best_loss_finite": true,
    "loss_decreased": true,
    "student_future_mse_finite": true,
    "teacher_future_mse_finite": true,
    "student_teacher_ratio_finite": true,
    "selected_top1_hit_rate": true,
    "selected_topk_hit_rate": true,
    "token_retention_ratio": true
  }
}
```

## Eval Summary

```json
{
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_structured_toy_v1/checkpoints/student_world_model_step_000300.pt",
  "selector_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_structured_toy_v1/checkpoints/student_selector_step_000200.pt",
  "selector_frozen": true,
  "dataset_size": 32,
  "device": "cuda",
  "topk": 4,
  "compressed_tokens": 8,
  "student_future_mse": 0.0023518727781871953,
  "prediction_mean": 0.00019946768588852137,
  "prediction_std": 0.5935211181640625,
  "target_mean": -0.00020117669191677123,
  "target_std": 0.591864287853241,
  "key_score_mean": 0.6109598278999329,
  "non_key_score_mean": 0.010969343595206738,
  "key_vs_non_key_gap": 0.5999904843047261,
  "top1_hit_rate": 1.0,
  "topk_hit_rate": 1.0,
  "selected_key_coverage": 1.0,
  "selected_key_fraction": 1.0,
  "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_structured_toy_v1/eval_summary.json"
}
```

## Teacher/Student Gap Summary

```json
{
  "student_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_structured_toy_v1/checkpoints/student_world_model_step_000300.pt",
  "teacher_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_structured_toy_tiny_v1/checkpoints/teacher_world_model_step_000100.pt",
  "dataset_size": 32,
  "device": "cuda",
  "teacher_future_mse": 0.002693264357124766,
  "student_future_mse": 0.0023518727781871953,
  "token_retention_ratio": 0.02040816326530612,
  "topk": 4,
  "total_tokens": 196,
  "compressed_tokens": 8,
  "student_teacher_gap": -0.00034139157893757054,
  "student_teacher_ratio": 0.8732424546315134,
  "key_score_mean": 0.6109598278999329,
  "non_key_score_mean": 0.010969343595206738,
  "key_vs_non_key_gap": 0.5999904843047261,
  "top1_hit_rate": 1.0,
  "topk_hit_rate": 1.0,
  "selected_key_coverage": 1.0,
  "selected_key_fraction": 1.0,
  "gap_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_structured_toy_v1/teacher_student_gap_summary.json"
}
```

## Checkpoint Inspection Summary

```json
{
  "step": 300,
  "selector_frozen": true,
  "compressor_config": {
    "token_dim": 768,
    "latent_dim": 128,
    "num_latents": 8,
    "hidden_dim": 256,
    "dropout": 0.0,
    "compressor_type": "perceiver_like",
    "num_heads": 8
  },
  "student_world_model_config": {
    "latent_dim": 128,
    "hidden_dim": 512,
    "output_dim": 768,
    "num_layers": 2,
    "dropout": 0.0,
    "pool": "mean"
  },
  "compressor_parameter_count": 298624,
  "student_world_model_parameter_count": 460288
}
```

STUDENT_WORLD_MODEL_TRAINING_SMOKE_PASS = true
