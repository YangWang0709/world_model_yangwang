# Student Selector BAIR VideoMAE Smoke Report

Command: `python scripts/smoke_test_student_selector_bair_videomae.py`

- train token shard dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train`
- test token shard dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test`
- train importance shard dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/train`
- test importance shard dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/test`
- run dir: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1`
- checkpoint path: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/checkpoints/student_selector_step_000100.pt`
- train samples: `100`
- test samples: `16`
- num tokens: `392`
- topk: `16`
- GPU memory before: `{'free_gib': 14.43, 'total_gib': 15.461, 'used_gib': 1.031}`
- GPU memory after: `{'free_gib': 14.324, 'total_gib': 15.461, 'used_gib': 1.136}`
- RAM before: `{'total_gib': 30.403, 'available_gib': 25.44, 'used_gib': 4.315}`
- RAM after: `{'total_gib': 30.403, 'available_gib': 24.511, 'used_gib': 5.228}`
- elapsed time sec: `0.817`
- OOM: `False`
- cloud recommendation: `not required for Step 11E; local RTX 5080 smoke succeeded`
- error: `None`

## Checks

```json
{
  "train_token_shards_exist": true,
  "test_token_shards_exist": true,
  "train_importance_shards_exist": true,
  "test_importance_shards_exist": true,
  "summary_exists": true,
  "metrics_exists": true,
  "checkpoint_exists": true,
  "num_steps_ok": true,
  "initial_loss_finite": true,
  "final_loss_finite": true,
  "best_loss_finite": true,
  "loss_decreased": true,
  "train_importance_mse_finite": true,
  "train_pearson_corr_finite": true,
  "train_target_topk_overlap_finite": true,
  "test_importance_mse_finite": true,
  "test_importance_mae_finite": true,
  "test_pearson_corr_finite": true,
  "test_pearson_corr_positive": true,
  "test_target_top1_overlap_finite": true,
  "test_target_topk_overlap_finite": true,
  "test_target_topk_overlap_nonzero": true,
  "test_selected_teacher_importance_finite": true,
  "test_random_teacher_importance_finite": true,
  "test_selected_vs_random_gap_finite": true,
  "test_selected_vs_random_gap_positive": true,
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
  "num_steps": 100,
  "initial_loss": 0.043159741908311844,
  "final_loss": 0.029176875948905945,
  "best_loss": 0.02254447154700756,
  "loss_decreased": true,
  "final_train_importance_mse": 0.03457484021782875,
  "final_train_importance_mae": 0.1508580893278122,
  "final_train_pearson_corr_mean": 0.3355245590209961,
  "final_train_target_top1_overlap": 0.009999999776482582,
  "final_train_target_topk_overlap": 0.16875,
  "final_train_selected_teacher_importance_mean": 0.6239902973175049,
  "final_train_random_teacher_importance_mean": 0.5122919678688049,
  "final_train_selected_vs_random_importance_gap": 0.11169832944869995,
  "test_importance_mse": 0.03200181573629379,
  "test_importance_mae": 0.14211484789848328,
  "test_pearson_corr_mean": 0.2978422939777374,
  "test_target_top1_overlap": 0.0,
  "test_target_topk_overlap": 0.1640625,
  "test_selected_teacher_importance_mean": 0.5927466750144958,
  "test_random_teacher_importance_mean": 0.4584925174713135,
  "test_selected_vs_random_importance_gap": 0.13425415754318237,
  "final_importance_mse": 0.03200181573629379,
  "final_importance_mae": 0.14211484789848328,
  "final_pearson_corr_mean": 0.2978422939777374,
  "final_target_top1_overlap": 0.0,
  "final_target_topk_overlap": 0.1640625,
  "final_selected_teacher_importance_mean": 0.5927466750144958,
  "final_random_teacher_importance_mean": 0.4584925174713135,
  "final_selected_vs_random_importance_gap": 0.13425415754318237,
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/checkpoints/student_selector_step_000100.pt",
  "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/metrics.jsonl",
  "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/summary.json",
  "device": "cuda",
  "dataset_size": 100,
  "train_dataset_size": 100,
  "test_dataset_size": 16,
  "num_samples": 16,
  "train_num_samples": 100,
  "test_num_samples": 16,
  "num_tokens": 392,
  "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1",
  "topk": 16,
  "train_eval_metrics": {
    "importance_mse": 0.03457484021782875,
    "importance_mae": 0.1508580893278122,
    "pearson_corr_mean": 0.3355245590209961,
    "target_top1_overlap": 0.009999999776482582,
    "target_topk_overlap": 0.16875,
    "selected_teacher_importance_mean": 0.6239902973175049,
    "random_teacher_importance_mean": 0.5122919678688049,
    "selected_vs_random_importance_gap": 0.11169832944869995,
    "score_mean": 0.4639049172401428,
    "score_std": 0.0680653303861618,
    "score_min": 0.18233636021614075,
    "score_max": 0.7337917685508728,
    "target_mean": 0.514139711856842,
    "target_std": 0.19880874454975128,
    "target_min": 0.0,
    "target_max": 1.0,
    "num_samples": 100,
    "num_tokens": 392,
    "topk": 16
  },
  "test_eval_metrics": {
    "importance_mse": 0.03200181573629379,
    "importance_mae": 0.14211484789848328,
    "pearson_corr_mean": 0.2978422939777374,
    "target_top1_overlap": 0.0,
    "target_topk_overlap": 0.1640625,
    "selected_teacher_importance_mean": 0.5927466750144958,
    "random_teacher_importance_mean": 0.4584925174713135,
    "selected_vs_random_importance_gap": 0.13425415754318237,
    "score_mean": 0.47237396240234375,
    "score_std": 0.06443193554878235,
    "score_min": 0.26193156838417053,
    "score_max": 0.6972493529319763,
    "target_mean": 0.4699629247188568,
    "target_std": 0.18634603917598724,
    "target_min": 0.0,
    "target_max": 1.0,
    "num_samples": 16,
    "num_tokens": 392,
    "topk": 16
  },
  "smoke_checks": {
    "train_token_shards_exist": true,
    "test_token_shards_exist": true,
    "train_importance_shards_exist": true,
    "test_importance_shards_exist": true,
    "summary_exists": true,
    "metrics_exists": true,
    "checkpoint_exists": true,
    "num_steps_ok": true,
    "initial_loss_finite": true,
    "final_loss_finite": true,
    "best_loss_finite": true,
    "loss_decreased": true,
    "train_importance_mse_finite": true,
    "train_pearson_corr_finite": true,
    "train_target_topk_overlap_finite": true,
    "test_importance_mse_finite": true,
    "test_importance_mae_finite": true,
    "test_pearson_corr_finite": true,
    "test_pearson_corr_positive": true,
    "test_target_top1_overlap_finite": true,
    "test_target_topk_overlap_finite": true,
    "test_target_topk_overlap_nonzero": true,
    "test_selected_teacher_importance_finite": true,
    "test_random_teacher_importance_finite": true,
    "test_selected_vs_random_gap_finite": true,
    "test_selected_vs_random_gap_positive": true,
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
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/checkpoints/student_selector_step_000100.pt",
  "split": "test",
  "dataset_size": 16,
  "token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test",
  "importance_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/test",
  "device": "cuda",
  "importance_mse": 0.03200181573629379,
  "importance_mae": 0.14211484789848328,
  "pearson_corr_mean": 0.2978422939777374,
  "target_top1_overlap": 0.0,
  "target_topk_overlap": 0.1640625,
  "selected_teacher_importance_mean": 0.5927466750144958,
  "random_teacher_importance_mean": 0.4584925174713135,
  "selected_vs_random_importance_gap": 0.13425415754318237,
  "score_mean": 0.47237396240234375,
  "score_std": 0.06443193554878235,
  "score_min": 0.26193156838417053,
  "score_max": 0.6972493529319763,
  "target_mean": 0.4699629247188568,
  "target_std": 0.18634603917598724,
  "target_min": 0.0,
  "target_max": 1.0,
  "num_samples": 16,
  "num_tokens": 392,
  "topk": 16,
  "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/eval_summary.json"
}
```

## Checkpoint Inspection

```json
{
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/checkpoints/student_selector_step_000100.pt",
  "checkpoint_keys": [
    "metrics_summary",
    "model_config",
    "model_state_dict",
    "optimizer_state_dict",
    "step"
  ],
  "model_config": {
    "token_dim": 768,
    "hidden_dim": 128,
    "task_dim": null,
    "use_task": false,
    "dropout": 0.0
  },
  "step": 100,
  "metrics_summary": {
    "num_steps": 100,
    "initial_loss": 0.043159741908311844,
    "final_loss": 0.029176875948905945,
    "best_loss": 0.02254447154700756,
    "loss_decreased": true,
    "final_train_importance_mse": 0.03457484021782875,
    "final_train_importance_mae": 0.1508580893278122,
    "final_train_pearson_corr_mean": 0.3355245590209961,
    "final_train_target_top1_overlap": 0.009999999776482582,
    "final_train_target_topk_overlap": 0.16875,
    "final_train_selected_teacher_importance_mean": 0.6239902973175049,
    "final_train_random_teacher_importance_mean": 0.5122919678688049,
    "final_train_selected_vs_random_importance_gap": 0.11169832944869995,
    "test_importance_mse": 0.03200181573629379,
    "test_importance_mae": 0.14211484789848328,
    "test_pearson_corr_mean": 0.2978422939777374,
    "test_target_top1_overlap": 0.0,
    "test_target_topk_overlap": 0.1640625,
    "test_selected_teacher_importance_mean": 0.5927466750144958,
    "test_random_teacher_importance_mean": 0.4584925174713135,
    "test_selected_vs_random_importance_gap": 0.13425415754318237,
    "final_importance_mse": 0.03200181573629379,
    "final_importance_mae": 0.14211484789848328,
    "final_pearson_corr_mean": 0.2978422939777374,
    "final_target_top1_overlap": 0.0,
    "final_target_topk_overlap": 0.1640625,
    "final_selected_teacher_importance_mean": 0.5927466750144958,
    "final_random_teacher_importance_mean": 0.4584925174713135,
    "final_selected_vs_random_importance_gap": 0.13425415754318237,
    "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/checkpoints/student_selector_step_000100.pt",
    "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/metrics.jsonl",
    "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/summary.json",
    "device": "cuda",
    "dataset_size": 100,
    "train_dataset_size": 100,
    "test_dataset_size": 16,
    "num_samples": 16,
    "train_num_samples": 100,
    "test_num_samples": 16,
    "num_tokens": 392,
    "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1",
    "topk": 16,
    "train_eval_metrics": {
      "importance_mse": 0.03457484021782875,
      "importance_mae": 0.1508580893278122,
      "pearson_corr_mean": 0.3355245590209961,
      "target_top1_overlap": 0.009999999776482582,
      "target_topk_overlap": 0.16875,
      "selected_teacher_importance_mean": 0.6239902973175049,
      "random_teacher_importance_mean": 0.5122919678688049,
      "selected_vs_random_importance_gap": 0.11169832944869995,
      "score_mean": 0.4639049172401428,
      "score_std": 0.0680653303861618,
      "score_min": 0.18233636021614075,
      "score_max": 0.7337917685508728,
      "target_mean": 0.514139711856842,
      "target_std": 0.19880874454975128,
      "target_min": 0.0,
      "target_max": 1.0,
      "num_samples": 100,
      "num_tokens": 392,
      "topk": 16
    },
    "test_eval_metrics": {
      "importance_mse": 0.03200181573629379,
      "importance_mae": 0.14211484789848328,
      "pearson_corr_mean": 0.2978422939777374,
      "target_top1_overlap": 0.0,
      "target_topk_overlap": 0.1640625,
      "selected_teacher_importance_mean": 0.5927466750144958,
      "random_teacher_importance_mean": 0.4584925174713135,
      "selected_vs_random_importance_gap": 0.13425415754318237,
      "score_mean": 0.47237396240234375,
      "score_std": 0.06443193554878235,
      "score_min": 0.26193156838417053,
      "score_max": 0.6972493529319763,
      "target_mean": 0.4699629247188568,
      "target_std": 0.18634603917598724,
      "target_min": 0.0,
      "target_max": 1.0,
      "num_samples": 16,
      "num_tokens": 392,
      "topk": 16
    }
  },
  "state_dict_key_count": 8,
  "parameter_count": 690689
}
```

STUDENT_SELECTOR_BAIR_VIDEOMAE_SMOKE_PASS = true
