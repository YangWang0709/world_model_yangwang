# Student World Model BAIR VideoMAE Smoke Report

Command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_student_world_model_bair_videomae.py`

- train token shard dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train`
- test token shard dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test`
- train importance shard dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/train`
- test importance shard dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/test`
- selector checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/checkpoints/student_selector_step_000100.pt`
- teacher checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1/checkpoints/teacher_world_model_step_000300.pt`
- run dir: `/home/ubuntu22/tgpawb_world_model/runs/student_world_model_bair_videomae_smoke_v1`
- checkpoint path: `/home/ubuntu22/tgpawb_world_model/runs/student_world_model_bair_videomae_smoke_v1/checkpoints/student_world_model_step_000500.pt`
- train samples: `100`
- test samples: `16`
- num tokens: `392`
- topk: `16`
- token_retention_ratio: `0.04081632653061224`
- elapsed_time_sec: `2.4545063972473145`
- oom: `false`
- cloud recommendation: `not required for Step 11F smoke`

## Training Summary

```json
{
  "num_steps": 500,
  "initial_loss": 12.400135040283203,
  "final_loss": 1.508751630783081,
  "best_loss": 1.0455163717269897,
  "loss_decreased": true,
  "final_future_loss": 1.508751630783081,
  "student_future_mse": 1.96855632464091,
  "final_student_future_mse": 1.96855632464091,
  "train_student_future_mse": 1.301594820022583,
  "train_selector_target_top1_overlap": 0.009999999776482582,
  "train_selector_target_topk_overlap": 0.16875,
  "train_selected_teacher_importance_mean": 0.6239902973175049,
  "train_random_teacher_importance_mean": 0.5122919678688049,
  "train_selected_vs_random_importance_gap": 0.11169832944869995,
  "test_student_future_mse": 1.96855632464091,
  "test_teacher_mse": 1.686881383260091,
  "test_teacher_future_mse": 1.686881383260091,
  "test_student_teacher_gap": 0.28167494138081883,
  "test_student_teacher_ratio": 1.1669796964837265,
  "test_selector_target_top1_overlap": 0.0,
  "test_selector_target_topk_overlap": 0.1640625,
  "test_selected_teacher_importance_mean": 0.5927466750144958,
  "test_random_teacher_importance_mean": 0.4584925174713135,
  "test_selected_vs_random_importance_gap": 0.13425415754318237,
  "final_selected_top1_hit_rate": null,
  "final_selected_topk_hit_rate": null,
  "final_selected_key_coverage": null,
  "final_selected_key_fraction": null,
  "final_selector_target_top1_overlap": 0.0,
  "final_selector_target_topk_overlap": 0.1640625,
  "final_selected_teacher_importance_mean": 0.5927466750144958,
  "final_random_teacher_importance_mean": 0.4584925174713135,
  "final_selected_vs_random_importance_gap": 0.13425415754318237,
  "token_retention_ratio": 0.04081632653061224,
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_bair_videomae_smoke_v1/checkpoints/student_world_model_step_000500.pt",
  "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_bair_videomae_smoke_v1/metrics.jsonl",
  "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_bair_videomae_smoke_v1/summary.json",
  "device": "cuda",
  "dataset_size": 100,
  "num_samples": 100,
  "train_num_samples": 100,
  "test_num_samples": 16,
  "num_tokens": 392,
  "token_dim": 768,
  "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_bair_videomae_smoke_v1",
  "selector_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/checkpoints/student_selector_step_000100.pt",
  "selector_checkpoint": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/checkpoints/student_selector_step_000100.pt",
  "selector_frozen": true,
  "teacher_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1/checkpoints/teacher_world_model_step_000300.pt",
  "topk": 16,
  "compressed_tokens": 16,
  "train_dataset_summary": {
    "split": "train",
    "dataset": "bair_robot_pushing_small",
    "num_samples": 100,
    "num_token_shards": 25,
    "num_importance_shards": 100,
    "num_tokens": 392,
    "token_dim": 768,
    "past_token_shape": [
      392,
      768
    ],
    "future_token_shape": [
      392,
      768
    ],
    "source_encoder": "unknown",
    "token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train",
    "importance_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/train",
    "first_token_shard_path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000000.pt",
    "first_importance_shard_path": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/train/importance_shard_000000.pt"
  },
  "test_dataset_summary": {
    "split": "test",
    "dataset": "bair_robot_pushing_small",
    "num_samples": 16,
    "num_token_shards": 4,
    "num_importance_shards": 16,
    "num_tokens": 392,
    "token_dim": 768,
    "past_token_shape": [
      392,
      768
    ],
    "future_token_shape": [
      392,
      768
    ],
    "source_encoder": "unknown",
    "token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test",
    "importance_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/test",
    "first_token_shard_path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test/tokens_shard_000000.pt",
    "first_importance_shard_path": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/test/importance_shard_000000.pt"
  },
  "train_token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train",
  "test_token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test",
  "train_importance_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/train",
  "test_importance_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/bair_videomae_teacher_smoke/test",
  "teacher_reference_eval": {
    "teacher_mse": 1.686881383260091,
    "teacher_future_mse": 1.686881383260091
  },
  "train_student_world_model_eval": {
    "student_future_mse": 1.301594820022583,
    "pred_mean": 0.29333245754241943,
    "pred_std": 3.2819361686706543,
    "pred_min": -6.566098690032959,
    "pred_max": 103.9301528930664,
    "prediction_mean": 0.29333245754241943,
    "prediction_std": 3.2819361686706543,
    "target_future_mean": 0.29530054330825806,
    "target_future_std": 3.5637941360473633,
    "target_future_min": -9.957594871520996,
    "target_future_max": 102.41267395019531,
    "target_mean": 0.514139711856842,
    "target_std": 0.19880874454975128,
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
    "target_min": 0.0,
    "target_max": 1.0,
    "selector_target_top1_overlap": 0.009999999776482582,
    "selector_target_topk_overlap": 0.16875,
    "target_importance_mean": 0.514139711856842,
    "target_importance_std": 0.19880874454975128,
    "target_importance_min": 0.0,
    "target_importance_max": 1.0
  },
  "student_world_model_eval": {
    "student_future_mse": 1.96855632464091,
    "pred_mean": 0.2880491614341736,
    "pred_std": 3.19828200340271,
    "pred_min": -5.909574031829834,
    "pred_max": 85.1462631225586,
    "prediction_mean": 0.2880491614341736,
    "prediction_std": 3.19828200340271,
    "target_future_mean": 0.3002580404281616,
    "target_future_std": 3.6612801551818848,
    "target_future_min": -9.533923149108887,
    "target_future_max": 96.4380874633789,
    "target_mean": 0.4699629247188568,
    "target_std": 0.18634603917598724,
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
    "target_min": 0.0,
    "target_max": 1.0,
    "selector_target_top1_overlap": 0.0,
    "selector_target_topk_overlap": 0.1640625,
    "target_importance_mean": 0.4699629247188568,
    "target_importance_std": 0.18634603917598724,
    "target_importance_min": 0.0,
    "target_importance_max": 1.0
  },
  "smoke_checks": {
    "summary_exists": true,
    "metrics_exists": true,
    "checkpoint_exists": true,
    "eval_summary_exists": true,
    "gap_summary_exists": true,
    "num_steps_ok": true,
    "initial_loss_finite": true,
    "final_loss_finite": true,
    "best_loss_finite": true,
    "student_future_mse_finite": true,
    "teacher_mse_finite": true,
    "student_teacher_ratio_finite": true,
    "selector_target_topk_overlap_finite": true,
    "selected_teacher_importance_mean_finite": true,
    "selected_vs_random_importance_gap_finite": true,
    "token_retention_ratio_ok": true,
    "selector_frozen": true,
    "ram_below_warning": true
  }
}
```

## Eval Summary

```json
{
  "split": "test",
  "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_bair_videomae_smoke_v1/checkpoints/student_world_model_step_000500.pt",
  "selector_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_smoke_v1/checkpoints/student_selector_step_000100.pt",
  "selector_frozen": true,
  "dataset_size": 16,
  "num_samples": 16,
  "device": "cuda",
  "topk": 16,
  "num_tokens": 392,
  "token_dim": 768,
  "token_retention_ratio": 0.04081632653061224,
  "compressed_tokens": 16,
  "student_future_mse": 1.96855632464091,
  "pred_mean": 0.2880491614341736,
  "pred_std": 3.19828200340271,
  "pred_min": -5.909574031829834,
  "pred_max": 85.1462631225586,
  "prediction_mean": 0.2880491614341736,
  "prediction_std": 3.19828200340271,
  "target_future_mean": 0.3002580404281616,
  "target_future_std": 3.6612801551818848,
  "target_future_min": -9.533923149108887,
  "target_future_max": 96.4380874633789,
  "target_mean": 0.4699629247188568,
  "target_std": 0.18634603917598724,
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
  "target_min": 0.0,
  "target_max": 1.0,
  "selector_target_top1_overlap": 0.0,
  "selector_target_topk_overlap": 0.1640625,
  "target_importance_mean": 0.4699629247188568,
  "target_importance_std": 0.18634603917598724,
  "target_importance_min": 0.0,
  "target_importance_max": 1.0,
  "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_bair_videomae_smoke_v1/eval_summary.json"
}
```

## Teacher-Student Gap Summary

```json
{
  "dataset": "BAIR",
  "split": "test",
  "student_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_bair_videomae_smoke_v1/checkpoints/student_world_model_step_000500.pt",
  "teacher_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_smoke_v1/checkpoints/teacher_world_model_step_000300.pt",
  "dataset_size": 16,
  "num_samples": 16,
  "device": "cuda",
  "teacher_future_mse": 1.686881383260091,
  "student_future_mse": 1.96855632464091,
  "teacher_mse": 1.686881383260091,
  "student_mse": 1.96855632464091,
  "token_retention_ratio": 0.04081632653061224,
  "topk": 16,
  "total_tokens": 392,
  "num_tokens": 392,
  "compressed_tokens": 16,
  "student_teacher_gap": 0.28167494138081883,
  "student_teacher_ratio": 1.1669796964837265,
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
  "selector_target_top1_overlap": 0.0,
  "selector_target_topk_overlap": 0.1640625,
  "target_importance_mean": 0.4699629247188568,
  "target_importance_std": 0.18634603917598724,
  "target_importance_min": 0.0,
  "target_importance_max": 1.0,
  "gap_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/student_world_model_bair_videomae_smoke_v1/teacher_student_gap_summary.json"
}
```

## Checkpoint Inspection Summary

```json
{
  "step": 500,
  "selector_frozen": true,
  "selector_config": {
    "token_dim": 768,
    "hidden_dim": 128,
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
  "selector_parameter_count": 690689,
  "compressor_parameter_count": 1453568,
  "student_world_model_parameter_count": 657664
}
```

## Resource Usage

```json
{
  "before": {
    "ram_used_gib": 6.3655242919921875,
    "ram_total_gib": 30.40261459350586,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.9208984375,
    "gpu_mem_used_gib": 0.7998046875
  },
  "after": {
    "ram_used_gib": 7.369956970214844,
    "ram_total_gib": 30.40261459350586,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.9208984375,
    "gpu_mem_used_gib": 1.2109375
  },
  "elapsed_time_sec": 2.4545063972473145,
  "max_ram_used_gib": 7.369956970214844,
  "max_gpu_mem_used_gib": 1.2109375,
  "oom": false,
  "gpu_mem_fraction": 0.07605962092866343,
  "cloud_recommendation": "not required for Step 11F smoke"
}
```

## Checks

```json
{
  "summary_exists": true,
  "metrics_exists": true,
  "checkpoint_exists": true,
  "eval_summary_exists": true,
  "gap_summary_exists": true,
  "num_steps_ok": true,
  "initial_loss_finite": true,
  "final_loss_finite": true,
  "best_loss_finite": true,
  "student_future_mse_finite": true,
  "teacher_mse_finite": true,
  "student_teacher_ratio_finite": true,
  "selector_target_topk_overlap_finite": true,
  "selected_teacher_importance_mean_finite": true,
  "selected_vs_random_importance_gap_finite": true,
  "token_retention_ratio_ok": true,
  "selector_frozen": true,
  "ram_below_warning": true
}
```

## Pytest Result

`79 passed in 1.41s`

STUDENT_WORLD_MODEL_BAIR_VIDEOMAE_SMOKE_PASS = true
