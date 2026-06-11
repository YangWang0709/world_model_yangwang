# Student World Model Real VideoMAE Smoke Report

Command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_student_world_model_real_video_videomae.py`

- token shard input dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke`
- importance shard input dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke`
- selector checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/checkpoints/student_selector_step_000300.pt`
- teacher checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/teacher_real_video_videomae_smoke_v1/checkpoints/teacher_world_model_step_000100.pt`
- run dir: `/home/ubuntu22/tgpawb_world_model/runs/student_world_model_real_video_videomae_smoke_v1`
- checkpoint path: `/home/ubuntu22/tgpawb_world_model/runs/student_world_model_real_video_videomae_smoke_v1/checkpoints/student_world_model_step_000500.pt`
- num samples: `8`
- num tokens: `784`
- topk: `16`
- token_retention_ratio: `0.02040816326530612`
- elapsed_time_sec: `2.0992462635040283`
- oom: `false`

## Training Summary

```json
{
  "num_steps": 500,
  "initial_loss": 23.0651912689209,
  "final_loss": 0.04984317347407341,
  "best_loss": 0.039038270711898804,
  "loss_decreased": true,
  "final_future_loss": 0.04984317347407341,
  "student_future_mse": 0.04321051016449928,
  "final_student_future_mse": 0.04321051016449928,
  "final_selected_top1_hit_rate": null,
  "final_selected_topk_hit_rate": null,
  "final_selected_key_coverage": null,
  "final_selected_key_fraction": null,
  "final_selector_target_top1_overlap": 0.125,
  "final_selector_target_topk_overlap": 0.3125,
  "final_selected_teacher_importance_mean": 0.6910883784294128,
  "final_random_teacher_importance_mean": 0.46076127886772156,
  "final_selected_vs_random_importance_gap": 0.23032709956169128,
  "token_retention_ratio": 0.02040816326530612,
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_real_video_videomae_smoke_v1/checkpoints/student_world_model_step_000500.pt",
  "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_real_video_videomae_smoke_v1/metrics.jsonl",
  "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_real_video_videomae_smoke_v1/summary.json",
  "device": "cuda",
  "dataset_size": 8,
  "num_samples": 8,
  "num_tokens": 784,
  "token_dim": 768,
  "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_real_video_videomae_smoke_v1",
  "selector_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/checkpoints/student_selector_step_000300.pt",
  "selector_checkpoint": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/checkpoints/student_selector_step_000300.pt",
  "selector_frozen": true,
  "topk": 16,
  "compressed_tokens": 16,
  "student_world_model_eval": {
    "student_future_mse": 0.04321051016449928,
    "pred_mean": 0.24964304268360138,
    "pred_std": 4.80546760559082,
    "pred_min": -10.6674222946167,
    "pred_max": 123.75480651855469,
    "prediction_mean": 0.24964304268360138,
    "prediction_std": 4.80546760559082,
    "target_future_mean": 0.25035932660102844,
    "target_future_std": 4.7856597900390625,
    "target_future_min": -11.218094825744629,
    "target_future_max": 123.22943878173828,
    "target_mean": 0.46076127886772156,
    "target_std": 0.33224743604660034,
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
    "target_min": 0.0,
    "target_max": 1.0,
    "selector_target_top1_overlap": 0.125,
    "selector_target_topk_overlap": 0.3125,
    "target_importance_mean": 0.46076127886772156,
    "target_importance_std": 0.33224743604660034,
    "target_importance_min": 0.0,
    "target_importance_max": 1.0,
    "selected_vs_random_importance_gap": 0.23032709956169128
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
    "teacher_mse_finite": true,
    "student_teacher_ratio_finite": true,
    "selector_target_topk_overlap_finite": true,
    "selected_teacher_importance_mean_finite": true,
    "token_retention_ratio_ok": true,
    "ram_below_warning": true
  }
}
```

## Eval Summary

```json
{
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_real_video_videomae_smoke_v1/checkpoints/student_world_model_step_000500.pt",
  "selector_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_real_video_videomae_smoke_v1/checkpoints/student_selector_step_000300.pt",
  "selector_frozen": true,
  "dataset_size": 8,
  "num_samples": 8,
  "device": "cuda",
  "topk": 16,
  "num_tokens": 784,
  "token_dim": 768,
  "token_retention_ratio": 0.02040816326530612,
  "compressed_tokens": 16,
  "student_future_mse": 0.04321051016449928,
  "pred_mean": 0.24964304268360138,
  "pred_std": 4.80546760559082,
  "pred_min": -10.6674222946167,
  "pred_max": 123.75480651855469,
  "prediction_mean": 0.24964304268360138,
  "prediction_std": 4.80546760559082,
  "target_future_mean": 0.25035932660102844,
  "target_future_std": 4.7856597900390625,
  "target_future_min": -11.218094825744629,
  "target_future_max": 123.22943878173828,
  "target_mean": 0.46076127886772156,
  "target_std": 0.33224743604660034,
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
  "target_min": 0.0,
  "target_max": 1.0,
  "selector_target_top1_overlap": 0.125,
  "selector_target_topk_overlap": 0.3125,
  "target_importance_mean": 0.46076127886772156,
  "target_importance_std": 0.33224743604660034,
  "target_importance_min": 0.0,
  "target_importance_max": 1.0,
  "selected_vs_random_importance_gap": 0.23032709956169128,
  "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_real_video_videomae_smoke_v1/eval_summary.json"
}
```

## Teacher-Student Gap Summary

```json
{
  "student_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_real_video_videomae_smoke_v1/checkpoints/student_world_model_step_000500.pt",
  "teacher_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_real_video_videomae_smoke_v1/checkpoints/teacher_world_model_step_000100.pt",
  "dataset_size": 8,
  "device": "cuda",
  "teacher_future_mse": 0.166172057390213,
  "student_future_mse": 0.04321051016449928,
  "teacher_mse": 0.166172057390213,
  "student_mse": 0.04321051016449928,
  "token_retention_ratio": 0.02040816326530612,
  "topk": 16,
  "total_tokens": 784,
  "compressed_tokens": 16,
  "student_teacher_gap": -0.12296154722571373,
  "student_teacher_ratio": 0.2600347545979427,
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
  "selector_target_top1_overlap": 0.125,
  "selector_target_topk_overlap": 0.3125,
  "target_importance_mean": 0.46076127886772156,
  "target_importance_std": 0.33224743604660034,
  "target_importance_min": 0.0,
  "target_importance_max": 1.0,
  "selected_vs_random_importance_gap": 0.23032709956169128,
  "gap_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_real_video_videomae_smoke_v1/teacher_student_gap_summary.json"
}
```

## Checkpoint Inspection Summary

```json
{
  "step": 500,
  "selector_frozen": true,
  "selector_config": {
    "token_dim": 768,
    "hidden_dim": 256,
    "task_dim": null,
    "use_task": false,
    "dropout": 0.0
  },
  "compressor_config": {
    "token_dim": 768,
    "latent_dim": 512,
    "num_latents": 16,
    "hidden_dim": 512,
    "dropout": 0.0,
    "compressor_type": "perceiver_like",
    "num_heads": 8
  },
  "student_world_model_config": {
    "latent_dim": 512,
    "hidden_dim": 512,
    "output_dim": 768,
    "num_layers": 2,
    "dropout": 0.0,
    "pool": "mean"
  },
  "selector_parameter_count": 789249,
  "compressor_parameter_count": 1453568,
  "student_world_model_parameter_count": 657664
}
```

## Resource Usage

```json
{
  "before": {
    "ram_used_gib": 3.5359230041503906,
    "ram_total_gib": 30.40261459350586,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.9208984375,
    "gpu_mem_used_gib": 0.8193359375
  },
  "after": {
    "ram_used_gib": 4.6351776123046875,
    "ram_total_gib": 30.40261459350586,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.9208984375,
    "gpu_mem_used_gib": 1.23046875
  },
  "elapsed_time_sec": 2.0992462635040283,
  "max_ram_used_gib": 4.6351776123046875,
  "max_gpu_mem_used_gib": 1.23046875,
  "oom": false,
  "cloud_recommendation": "not required for this 8-sample smoke"
}
```

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
  "student_future_mse_finite": true,
  "teacher_mse_finite": true,
  "student_teacher_ratio_finite": true,
  "selector_target_topk_overlap_finite": true,
  "selected_teacher_importance_mean_finite": true,
  "token_retention_ratio_ok": true,
  "ram_below_warning": true
}
```

STUDENT_WORLD_MODEL_REAL_VIDEOMAE_SMOKE_PASS = true
