# Student Selector Training Smoke Report

Command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_student_selector_training.py`

- token shard input dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/structured_toy`
- importance shard input dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/structured_toy_teacher`
- run dir: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_structured_toy_v1`
- checkpoint path: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_structured_toy_v1/checkpoints/student_selector_step_000200.pt`
- num steps: `200`
- initial loss: `0.26841574907302856`
- final loss: `0.000204412717721425`
- best loss: `0.000204412717721425`
- loss_decreased: `True`
- final key score mean: `0.6109598278999329`
- final non-key score mean: `0.010969343595206738`
- final key_vs_non_key_gap: `0.5999904843047261`
- final top1 hit rate: `1.0`
- final topk hit rate: `1.0`

## Training Summary

```json
{
  "num_steps": 200,
  "initial_loss": 0.26841574907302856,
  "final_loss": 0.000204412717721425,
  "best_loss": 0.000204412717721425,
  "loss_decreased": true,
  "final_key_score_mean": 0.6109598278999329,
  "final_non_key_score_mean": 0.010969343595206738,
  "final_key_vs_non_key_gap": 0.5999904843047261,
  "final_top1_hit_rate": 1.0,
  "final_topk_hit_rate": 1.0,
  "final_importance_mse": 0.00022361590526998043,
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_structured_toy_v1/checkpoints/student_selector_step_000200.pt",
  "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_structured_toy_v1/metrics.jsonl",
  "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_structured_toy_v1/summary.json",
  "device": "cuda",
  "dataset_size": 32,
  "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_structured_toy_v1",
  "topk": 4,
  "smoke_checks": {
    "summary_exists": true,
    "metrics_exists": true,
    "checkpoint_exists": true,
    "num_steps_ok": true,
    "initial_loss_finite": true,
    "final_loss_finite": true,
    "best_loss_finite": true,
    "loss_decreased": true,
    "key_gt_non_key": true,
    "top1_hit_rate": true,
    "topk_hit_rate": true
  }
}
```

## Eval Summary

```json
{
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_structured_toy_v1/checkpoints/student_selector_step_000200.pt",
  "dataset_size": 32,
  "device": "cuda",
  "importance_mse": 0.00022361590526998043,
  "score_mean": 0.02321404591202736,
  "score_std": 0.09119120240211487,
  "score_min": 0.00039221972110681236,
  "score_max": 0.9643838405609131,
  "target_mean": 0.012825509533286095,
  "target_std": 0.09571697562932968,
  "target_min": 0.0,
  "target_max": 1.0,
  "key_score_mean": 0.6109598278999329,
  "non_key_score_mean": 0.010969343595206738,
  "key_vs_non_key_gap": 0.5999904843047261,
  "top1_hit_rate": 1.0,
  "topk_hit_rate": 1.0,
  "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_structured_toy_v1/eval_summary.json"
}
```

## Checkpoint Inspection Summary

```json
{
  "step": 200,
  "model_config": {
    "token_dim": 768,
    "hidden_dim": 256,
    "task_dim": null,
    "use_task": false,
    "dropout": 0.0
  },
  "parameter_count": 789249,
  "state_dict_key_count": 8
}
```

STUDENT_SELECTOR_TRAINING_SMOKE_PASS = true
