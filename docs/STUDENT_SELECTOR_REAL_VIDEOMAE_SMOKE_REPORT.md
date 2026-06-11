# Student Selector Real VideoMAE Smoke Report

Command: `python scripts/smoke_test_student_selector_real_video_videomae.py`

- token shard input dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke`
- importance shard input dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke`
- run dir: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1`
- checkpoint path: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/checkpoints/student_selector_step_000300.pt`
- num samples: `8`
- num tokens: `784`
- topk: `16`
- GPU memory before: `{'free_gib': 14.512, 'total_gib': 15.461, 'used_gib': 0.948}`
- GPU memory after: `{'free_gib': 14.407, 'total_gib': 15.461, 'used_gib': 1.054}`
- RAM before: `{'total_gib': 30.403, 'available_gib': 25.72, 'used_gib': 4.202}`
- RAM after: `{'total_gib': 30.403, 'available_gib': 24.895, 'used_gib': 5.012}`
- elapsed time sec: `0.984`
- OOM: `False`
- cloud recommendation: `not required for Step 10C; local RTX 5080 smoke succeeded`
- error: `None`

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
  "loss_decreased_or_warn": true,
  "final_importance_mse_finite": true,
  "final_importance_mae_finite": true,
  "final_pearson_corr_finite": true,
  "final_target_topk_overlap_finite": true,
  "final_target_topk_overlap_positive": true,
  "eval_importance_mse_finite": true,
  "eval_target_topk_overlap_finite": true,
  "parameter_count_ok": true,
  "oom_ok": true,
  "resource_warning_ok": true
}
```

## Training Summary

```json
{
  "num_steps": 300,
  "initial_loss": 0.09144660085439682,
  "final_loss": 0.01454525999724865,
  "best_loss": 0.006556759588420391,
  "loss_decreased": true,
  "final_importance_mse": 0.018842598423361778,
  "final_importance_mae": 0.10066770762205124,
  "final_pearson_corr_mean": 0.5616448521614075,
  "final_target_top1_overlap": 0.125,
  "final_target_topk_overlap": 0.3125,
  "final_selected_teacher_importance_mean": 0.6910883784294128,
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/checkpoints/student_selector_step_000300.pt",
  "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/metrics.jsonl",
  "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/summary.json",
  "device": "cuda",
  "dataset_size": 8,
  "num_samples": 8,
  "num_tokens": 784,
  "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1",
  "topk": 16,
  "smoke_checks": {
    "summary_exists": true,
    "metrics_exists": true,
    "checkpoint_exists": true,
    "num_steps_ok": true,
    "initial_loss_finite": true,
    "final_loss_finite": true,
    "best_loss_finite": true,
    "loss_decreased": true,
    "loss_decreased_or_warn": true,
    "final_importance_mse_finite": true,
    "final_importance_mae_finite": true,
    "final_pearson_corr_finite": true,
    "final_target_topk_overlap_finite": true,
    "final_target_topk_overlap_positive": true,
    "eval_importance_mse_finite": true,
    "eval_target_topk_overlap_finite": true,
    "parameter_count_ok": true,
    "oom_ok": true,
    "resource_warning_ok": true
  }
}
```

## Eval Summary

```json
{
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/checkpoints/student_selector_step_000300.pt",
  "dataset_size": 8,
  "device": "cuda",
  "importance_mse": 0.018842598423361778,
  "importance_mae": 0.10066770762205124,
  "pearson_corr_mean": 0.5616448521614075,
  "target_top1_overlap": 0.125,
  "target_topk_overlap": 0.3125,
  "selected_teacher_importance_mean": 0.6910883784294128,
  "random_teacher_importance_mean": 0.46076127886772156,
  "score_mean": 0.42939504981040955,
  "score_std": 0.28723156452178955,
  "score_min": 0.006375065539032221,
  "score_max": 0.9908421039581299,
  "target_mean": 0.46076127886772156,
  "target_std": 0.33224743604660034,
  "target_min": 0.0,
  "target_max": 1.0,
  "num_samples": 8,
  "num_tokens": 784,
  "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/eval_summary.json"
}
```

## Checkpoint Inspection

```json
{
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/checkpoints/student_selector_step_000300.pt",
  "checkpoint_keys": [
    "metrics_summary",
    "model_config",
    "model_state_dict",
    "optimizer_state_dict",
    "step"
  ],
  "model_config": {
    "token_dim": 768,
    "hidden_dim": 256,
    "task_dim": null,
    "use_task": false,
    "dropout": 0.0
  },
  "step": 300,
  "metrics_summary": {
    "num_steps": 300,
    "initial_loss": 0.09144660085439682,
    "final_loss": 0.01454525999724865,
    "best_loss": 0.006556759588420391,
    "loss_decreased": true,
    "final_importance_mse": 0.018842598423361778,
    "final_importance_mae": 0.10066770762205124,
    "final_pearson_corr_mean": 0.5616448521614075,
    "final_target_top1_overlap": 0.125,
    "final_target_topk_overlap": 0.3125,
    "final_selected_teacher_importance_mean": 0.6910883784294128,
    "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/checkpoints/student_selector_step_000300.pt",
    "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/metrics.jsonl",
    "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/summary.json",
    "device": "cuda",
    "dataset_size": 8,
    "num_samples": 8,
    "num_tokens": 784,
    "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1",
    "topk": 16
  },
  "state_dict_key_count": 8,
  "parameter_count": 789249
}
```

STUDENT_SELECTOR_REAL_VIDEOMAE_SMOKE_PASS = true
