# BAIR Selector Ablation Smoke Report

Command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_bair_selector_ablation.py`

- run dir: `/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1`
- selector variants: `mse_only, weighted_mse_alpha2, mse_rank_w0p1, topk_bce, hybrid_weighted_mse_rank_bce`
- selector summary: `/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selector_ablation_summary.json`
- downstream summary: `/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/downstream_ablation_summary.json`
- combined report: `/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selector_ablation_combined_report.json`

## Selector-Level Summary

| variant | final_loss | importance_mse | pearson_corr | top1_overlap | topk_overlap | selected_teacher_importance_mean | selected_vs_random_importance_gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid_weighted_mse_rank_bce | 0.23140690 | 0.05999130 | -0.060447 | 0.125000 | 0.144531 | 0.441189 | -0.017304 |
| mse_only | 0.02415186 | 0.04716463 | -0.072140 | 0.000000 | 0.023438 | 0.479386 | 0.020893 |
| mse_rank_w0p1 | 0.05464894 | 0.05608696 | -0.023212 | 0.000000 | 0.113281 | 0.468579 | 0.010086 |
| topk_bce | 0.54983288 | 0.18720281 | 0.013806 | 0.000000 | 0.183594 | 0.436488 | -0.022004 |
| weighted_mse_alpha2 | 0.04349550 | 0.04610213 | -0.011240 | 0.000000 | 0.050781 | 0.512385 | 0.053892 |

## Selector Aggregate

```json
{
  "num_variants": 5,
  "num_successful": 5,
  "failed_variants": [],
  "best_by_selected_teacher_importance": {
    "variant_name": "weighted_mse_alpha2",
    "variant_run_name": "weighted_mse_alpha2_seed0",
    "loss_type": "weighted_mse",
    "seed": 0,
    "success": true,
    "num_steps": 300,
    "initial_loss": 0.09764230996370316,
    "final_loss": 0.043495502322912216,
    "best_loss": 0.016780130565166473,
    "loss_decreased": true,
    "test_importance_mse": 0.0461021289229393,
    "test_importance_mae": 0.17089954018592834,
    "test_pearson_corr_mean": -0.011240443214774132,
    "test_target_top1_overlap": 0.0,
    "test_target_topk_overlap": 0.05078125,
    "test_selected_teacher_importance_mean": 0.512384831905365,
    "test_random_teacher_importance_mean": 0.4584925174713135,
    "test_selected_vs_random_importance_gap": 0.053892314434051514,
    "score_mean": 0.5207911133766174,
    "score_std": 0.10554179549217224,
    "target_mean": 0.4699629247188568,
    "target_std": 0.18634603917598724,
    "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/checkpoints/student_selector_step_000300.pt",
    "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/metrics.jsonl",
    "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/summary.json",
    "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/eval_summary.json",
    "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0",
    "device": "cuda",
    "topk": 16,
    "train_num_samples": 100,
    "test_num_samples": 16,
    "num_tokens": 392,
    "token_dim": 768,
    "loss_config": {
      "name": "weighted_mse_alpha2",
      "type": "weighted_mse",
      "seed": 0,
      "alpha": 2.0,
      "topk": 16
    },
    "train_eval_metrics": {
      "importance_mse": 0.017144901677966118,
      "importance_mae": 0.1000242531299591,
      "pearson_corr_mean": 0.6580300331115723,
      "target_top1_overlap": 0.07999999821186066,
      "target_topk_overlap": 0.295625,
      "selected_teacher_importance_mean": 0.7368444800376892,
      "random_teacher_importance_mean": 0.5122919678688049,
      "selected_vs_random_importance_gap": 0.22455251216888428,
      "score_mean": 0.5034849047660828,
      "score_std": 0.14014092087745667,
      "score_min": 0.06424645334482193,
      "score_max": 0.9426285624504089,
      "target_mean": 0.514139711856842,
      "target_std": 0.19880874454975128,
      "target_min": 0.0,
      "target_max": 1.0,
      "num_samples": 100,
      "num_tokens": 392,
      "topk": 16
    },
    "test_eval_metrics": {
      "importance_mse": 0.0461021289229393,
      "importance_mae": 0.17089954018592834,
      "pearson_corr_mean": -0.011240443214774132,
      "target_top1_overlap": 0.0,
      "target_topk_overlap": 0.05078125,
      "selected_teacher_importance_mean": 0.512384831905365,
      "random_teacher_importance_mean": 0.4584925174713135,
      "selected_vs_random_importance_gap": 0.053892314434051514,
      "score_mean": 0.5207911133766174,
      "score_std": 0.10554179549217224,
      "score_min": 0.19795392453670502,
      "score_max": 0.9451943039894104,
      "target_mean": 0.4699629247188568,
      "target_std": 0.18634603917598724,
      "target_min": 0.0,
      "target_max": 1.0,
      "num_samples": 16,
      "num_tokens": 392,
      "topk": 16
    }
  },
  "best_by_target_topk_overlap": {
    "variant_name": "topk_bce",
    "variant_run_name": "topk_bce_seed0",
    "loss_type": "topk_bce",
    "seed": 0,
    "success": true,
    "num_steps": 300,
    "initial_loss": 1.3922330141067505,
    "final_loss": 0.5498328804969788,
    "best_loss": 0.3215303122997284,
    "loss_decreased": true,
    "test_importance_mse": 0.1872028112411499,
    "test_importance_mae": 0.388860821723938,
    "test_pearson_corr_mean": 0.01380588673055172,
    "test_target_top1_overlap": 0.0,
    "test_target_topk_overlap": 0.18359375,
    "test_selected_teacher_importance_mean": 0.4364883303642273,
    "test_random_teacher_importance_mean": 0.4584925174713135,
    "test_selected_vs_random_importance_gap": -0.02200418710708618,
    "score_mean": 0.16488496959209442,
    "score_std": 0.2521185576915741,
    "target_mean": 0.4699629247188568,
    "target_std": 0.18634603917598724,
    "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/topk_bce_seed0/checkpoints/student_selector_step_000300.pt",
    "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/topk_bce_seed0/metrics.jsonl",
    "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/topk_bce_seed0/summary.json",
    "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/topk_bce_seed0/eval_summary.json",
    "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/topk_bce_seed0",
    "device": "cuda",
    "topk": 16,
    "train_num_samples": 100,
    "test_num_samples": 16,
    "num_tokens": 392,
    "token_dim": 768,
    "loss_config": {
      "name": "topk_bce",
      "type": "topk_bce",
      "seed": 0,
      "bce_loss_weight": 1.0,
      "topk": 16
    },
    "train_eval_metrics": {
      "importance_mse": 0.2014135718345642,
      "importance_mae": 0.4039130210876465,
      "pearson_corr_mean": 0.33517196774482727,
      "target_top1_overlap": 0.15000000596046448,
      "target_topk_overlap": 0.543125,
      "selected_teacher_importance_mean": 0.6879673600196838,
      "random_teacher_importance_mean": 0.5122919678688049,
      "selected_vs_random_importance_gap": 0.1756753921508789,
      "score_mean": 0.1548847258090973,
      "score_std": 0.25551918148994446,
      "score_min": 8.26540524911934e-09,
      "score_max": 0.9999405145645142,
      "target_mean": 0.514139711856842,
      "target_std": 0.19880874454975128,
      "target_min": 0.0,
      "target_max": 1.0,
      "num_samples": 100,
      "num_tokens": 392,
      "topk": 16
    },
    "test_eval_metrics": {
      "importance_mse": 0.1872028112411499,
      "importance_mae": 0.388860821723938,
      "pearson_corr_mean": 0.01380588673055172,
      "target_top1_overlap": 0.0,
      "target_topk_overlap": 0.18359375,
      "selected_teacher_importance_mean": 0.4364883303642273,
      "random_teacher_importance_mean": 0.4584925174713135,
      "selected_vs_random_importance_gap": -0.02200418710708618,
      "score_mean": 0.16488496959209442,
      "score_std": 0.2521185576915741,
      "score_min": 5.1745423945703806e-08,
      "score_max": 0.9990449547767639,
      "target_mean": 0.4699629247188568,
      "target_std": 0.18634603917598724,
      "target_min": 0.0,
      "target_max": 1.0,
      "num_samples": 16,
      "num_tokens": 392,
      "topk": 16
    }
  },
  "best_by_importance_mse": {
    "variant_name": "weighted_mse_alpha2",
    "variant_run_name": "weighted_mse_alpha2_seed0",
    "loss_type": "weighted_mse",
    "seed": 0,
    "success": true,
    "num_steps": 300,
    "initial_loss": 0.09764230996370316,
    "final_loss": 0.043495502322912216,
    "best_loss": 0.016780130565166473,
    "loss_decreased": true,
    "test_importance_mse": 0.0461021289229393,
    "test_importance_mae": 0.17089954018592834,
    "test_pearson_corr_mean": -0.011240443214774132,
    "test_target_top1_overlap": 0.0,
    "test_target_topk_overlap": 0.05078125,
    "test_selected_teacher_importance_mean": 0.512384831905365,
    "test_random_teacher_importance_mean": 0.4584925174713135,
    "test_selected_vs_random_importance_gap": 0.053892314434051514,
    "score_mean": 0.5207911133766174,
    "score_std": 0.10554179549217224,
    "target_mean": 0.4699629247188568,
    "target_std": 0.18634603917598724,
    "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/checkpoints/student_selector_step_000300.pt",
    "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/metrics.jsonl",
    "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/summary.json",
    "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/eval_summary.json",
    "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0",
    "device": "cuda",
    "topk": 16,
    "train_num_samples": 100,
    "test_num_samples": 16,
    "num_tokens": 392,
    "token_dim": 768,
    "loss_config": {
      "name": "weighted_mse_alpha2",
      "type": "weighted_mse",
      "seed": 0,
      "alpha": 2.0,
      "topk": 16
    },
    "train_eval_metrics": {
      "importance_mse": 0.017144901677966118,
      "importance_mae": 0.1000242531299591,
      "pearson_corr_mean": 0.6580300331115723,
      "target_top1_overlap": 0.07999999821186066,
      "target_topk_overlap": 0.295625,
      "selected_teacher_importance_mean": 0.7368444800376892,
      "random_teacher_importance_mean": 0.5122919678688049,
      "selected_vs_random_importance_gap": 0.22455251216888428,
      "score_mean": 0.5034849047660828,
      "score_std": 0.14014092087745667,
      "score_min": 0.06424645334482193,
      "score_max": 0.9426285624504089,
      "target_mean": 0.514139711856842,
      "target_std": 0.19880874454975128,
      "target_min": 0.0,
      "target_max": 1.0,
      "num_samples": 100,
      "num_tokens": 392,
      "topk": 16
    },
    "test_eval_metrics": {
      "importance_mse": 0.0461021289229393,
      "importance_mae": 0.17089954018592834,
      "pearson_corr_mean": -0.011240443214774132,
      "target_top1_overlap": 0.0,
      "target_topk_overlap": 0.05078125,
      "selected_teacher_importance_mean": 0.512384831905365,
      "random_teacher_importance_mean": 0.4584925174713135,
      "selected_vs_random_importance_gap": 0.053892314434051514,
      "score_mean": 0.5207911133766174,
      "score_std": 0.10554179549217224,
      "score_min": 0.19795392453670502,
      "score_max": 0.9451943039894104,
      "target_mean": 0.4699629247188568,
      "target_std": 0.18634603917598724,
      "target_min": 0.0,
      "target_max": 1.0,
      "num_samples": 16,
      "num_tokens": 392,
      "topk": 16
    }
  },
  "mean_selected_teacher_importance": 0.46760536432266236,
  "mean_target_topk_overlap": 0.103125,
  "sanity_gate": {
    "checks": {
      "selector_metrics_finite": true,
      "mse_only_variant_success": true,
      "non_mse_only_variant_success": true,
      "best_selector_importance_finite": true
    },
    "pass": true
  }
}
```


## Downstream-Level Summary

| variant | student_future_mse | teacher_mse | student_teacher_ratio | token_retention_ratio | selector_target_topk_overlap | selected_teacher_importance_mean | selected_vs_random_importance_gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid_weighted_mse_rank_bce | 1.83763584 | 1.68688138 | 1.089369 | 0.040816 | 0.144531 | 0.441189 | -0.017304 |
| mse_only | 1.83880289 | 1.68688138 | 1.090061 | 0.040816 | 0.023438 | 0.479386 | 0.020893 |
| mse_rank_w0p1 | 1.94753659 | 1.68688138 | 1.154519 | 0.040816 | 0.113281 | 0.468579 | 0.010086 |
| topk_bce | 1.82720673 | 1.68688138 | 1.083186 | 0.040816 | 0.183594 | 0.436488 | -0.022004 |
| weighted_mse_alpha2 | 1.76074374 | 1.68688138 | 1.043786 | 0.040816 | 0.050781 | 0.512385 | 0.053892 |

## Downstream Aggregate

```json
{
  "num_variants": 5,
  "num_successful": 5,
  "failed_variants": [],
  "best_by_student_future_mse": {
    "policy": "learned_selector",
    "seed": 0,
    "num_steps": 500,
    "initial_loss": 12.982955932617188,
    "final_loss": 1.3612430095672607,
    "best_loss": 0.9276008009910583,
    "loss_decreased": true,
    "student_future_mse": 1.7607437372207642,
    "student_mse": 1.7607437372207642,
    "teacher_future_mse": 1.686881383260091,
    "teacher_mse": 1.686881383260091,
    "student_teacher_gap": 0.07386235396067309,
    "student_teacher_ratio": 1.0437863353604186,
    "token_retention_ratio": 0.04081632653061224,
    "selector_target_top1_overlap": 0.0,
    "selector_target_topk_overlap": 0.05078125,
    "selected_teacher_importance_mean": 0.512384831905365,
    "random_teacher_importance_mean": 0.4584925174713135,
    "selected_vs_random_importance_gap": 0.053892314434051514,
    "selected_top1_hit_rate": null,
    "selected_topk_hit_rate": null,
    "selected_key_coverage": null,
    "selected_key_fraction": null,
    "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/downstream_student_world_models/weighted_mse_alpha2_seed0/checkpoints/student_world_model_step_000500.pt",
    "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/downstream_student_world_models/weighted_mse_alpha2_seed0/metrics.jsonl",
    "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/downstream_student_world_models/weighted_mse_alpha2_seed0/summary.json",
    "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/downstream_student_world_models/weighted_mse_alpha2_seed0/eval_summary.json",
    "gap_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/downstream_student_world_models/weighted_mse_alpha2_seed0/teacher_student_gap_summary.json",
    "device": "cuda",
    "dataset_size": 100,
    "train_num_samples": 100,
    "test_num_samples": 16,
    "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/downstream_student_world_models/weighted_mse_alpha2_seed0",
    "topk": 16,
    "total_tokens": 392,
    "token_dim": 768,
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
    "variant_name": "weighted_mse_alpha2",
    "variant_run_name": "weighted_mse_alpha2_seed0",
    "loss_type": "weighted_mse",
    "selector_checkpoint": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/checkpoints/student_selector_step_000300.pt",
    "selector_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/checkpoints/student_selector_step_000300.pt",
    "success": true
  },
  "step12_reference": {
    "summary_json": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/baseline_summary.json",
    "random_k_mean": {
      "method": "step12_random_k_mean",
      "student_future_mse": 1.8781254159079657,
      "selected_teacher_importance_mean": 0.4708937505880992,
      "selector_target_topk_overlap": 0.036458333333333336
    },
    "uniform_k": {
      "policy": "uniform_k",
      "seed": 0,
      "num_steps": 500,
      "initial_loss": 13.027299880981445,
      "final_loss": 1.365092158317566,
      "best_loss": 0.9896382093429565,
      "loss_decreased": true,
      "student_future_mse": 1.8350162506103516,
      "student_mse": 1.8350162506103516,
      "teacher_future_mse": 1.686881383260091,
      "teacher_mse": 1.686881383260091,
      "student_teacher_gap": 0.1481348673502605,
      "student_teacher_ratio": 1.0878158172947365,
      "token_retention_ratio": 0.04081632653061224,
      "selector_target_top1_overlap": 0.0,
      "selector_target_topk_overlap": 0.0546875,
      "selected_teacher_importance_mean": 0.48264437913894653,
      "random_teacher_importance_mean": 0.4584925174713135,
      "selected_vs_random_importance_gap": 0.024151861667633057,
      "selected_top1_hit_rate": null,
      "selected_topk_hit_rate": null,
      "selected_key_coverage": null,
      "selected_key_fraction": null,
      "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/uniform_k_seed0/checkpoints/student_world_model_step_000500.pt",
      "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/uniform_k_seed0/metrics.jsonl",
      "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/uniform_k_seed0/summary.json",
      "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/uniform_k_seed0/eval_summary.json",
      "gap_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/uniform_k_seed0/teacher_student_gap_summary.json",
      "device": "cuda",
      "dataset_size": 100,
      "train_num_samples": 100,
      "test_num_samples": 16,
      "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/uniform_k_seed0",
      "topk": 16,
      "total_tokens": 392,
      "token_dim": 768,
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
      }
    },
    "teacher_importance_topk": {
      "policy": "teacher_importance_topk",
      "seed": 0,
      "num_steps": 500,
      "initial_loss": 13.014167785644531,
      "final_loss": 1.3775962591171265,
      "best_loss": 1.0255603790283203,
      "loss_decreased": true,
      "student_future_mse": 1.760870138804118,
      "student_mse": 1.760870138804118,
      "teacher_future_mse": 1.686881383260091,
      "teacher_mse": 1.686881383260091,
      "student_teacher_gap": 0.07398875554402684,
      "student_teacher_ratio": 1.0438612674715961,
      "token_retention_ratio": 0.04081632653061224,
      "selector_target_top1_overlap": 1.0,
      "selector_target_topk_overlap": 0.99609375,
      "selected_teacher_importance_mean": 0.835728645324707,
      "random_teacher_importance_mean": 0.4584925174713135,
      "selected_vs_random_importance_gap": 0.37723612785339355,
      "selected_top1_hit_rate": null,
      "selected_topk_hit_rate": null,
      "selected_key_coverage": null,
      "selected_key_fraction": null,
      "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/teacher_importance_topk_seed0/checkpoints/student_world_model_step_000500.pt",
      "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/teacher_importance_topk_seed0/metrics.jsonl",
      "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/teacher_importance_topk_seed0/summary.json",
      "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/teacher_importance_topk_seed0/eval_summary.json",
      "gap_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/teacher_importance_topk_seed0/teacher_student_gap_summary.json",
      "device": "cuda",
      "dataset_size": 100,
      "train_num_samples": 100,
      "test_num_samples": 16,
      "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/teacher_importance_topk_seed0",
      "topk": 16,
      "total_tokens": 392,
      "token_dim": 768,
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
      }
    },
    "learned_selector": {
      "policy": "learned_selector",
      "seed": 0,
      "num_steps": 500,
      "initial_loss": 12.903804779052734,
      "final_loss": 1.3812907934188843,
      "best_loss": 1.0424336194992065,
      "loss_decreased": true,
      "student_future_mse": 1.8413029511769612,
      "student_mse": 1.8413029511769612,
      "teacher_future_mse": 1.686881383260091,
      "teacher_mse": 1.686881383260091,
      "student_teacher_gap": 0.15442156791687012,
      "student_teacher_ratio": 1.0915426356881317,
      "token_retention_ratio": 0.04081632653061224,
      "selector_target_top1_overlap": 0.25,
      "selector_target_topk_overlap": 0.1640625,
      "selected_teacher_importance_mean": 0.5927466750144958,
      "random_teacher_importance_mean": 0.4584925174713135,
      "selected_vs_random_importance_gap": 0.13425415754318237,
      "selected_top1_hit_rate": null,
      "selected_topk_hit_rate": null,
      "selected_key_coverage": null,
      "selected_key_fraction": null,
      "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/learned_selector_seed0/checkpoints/student_world_model_step_000500.pt",
      "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/learned_selector_seed0/metrics.jsonl",
      "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/learned_selector_seed0/summary.json",
      "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/learned_selector_seed0/eval_summary.json",
      "gap_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/learned_selector_seed0/teacher_student_gap_summary.json",
      "device": "cuda",
      "dataset_size": 100,
      "train_num_samples": 100,
      "test_num_samples": 16,
      "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/learned_selector_seed0",
      "topk": 16,
      "total_tokens": 392,
      "token_dim": 768,
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
      }
    }
  },
  "comparison": {
    "downstream_improvement_over_step12_learned": true,
    "best_downstream_vs_step12_learned_mse_delta": -0.08055921395619703,
    "any_variant_selected_importance_gt_step12_learned": false,
    "any_variant_topk_overlap_gt_step12_learned": true,
    "best_downstream_beats_step12_uniform_mse": true,
    "best_downstream_vs_step12_uniform_mse_delta": -0.0742725133895874,
    "best_downstream_gap_to_teacher_importance_topk_mse": -0.0001264015833537524
  },
  "sanity_gate": {
    "checks": {
      "downstream_result_present": true,
      "downstream_metrics_finite": true,
      "best_downstream_mse_finite": true
    },
    "pass": true
  }
}
```


## Comparison With Step 12

# BAIR Selector Ablation Combined Report

## Selector-Level Results

| variant | final_loss | importance_mse | pearson_corr | top1_overlap | topk_overlap | selected_teacher_importance_mean | selected_vs_random_importance_gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid_weighted_mse_rank_bce | 0.23140690 | 0.05999130 | -0.060447 | 0.125000 | 0.144531 | 0.441189 | -0.017304 |
| mse_only | 0.02415186 | 0.04716463 | -0.072140 | 0.000000 | 0.023438 | 0.479386 | 0.020893 |
| mse_rank_w0p1 | 0.05464894 | 0.05608696 | -0.023212 | 0.000000 | 0.113281 | 0.468579 | 0.010086 |
| topk_bce | 0.54983288 | 0.18720281 | 0.013806 | 0.000000 | 0.183594 | 0.436488 | -0.022004 |
| weighted_mse_alpha2 | 0.04349550 | 0.04610213 | -0.011240 | 0.000000 | 0.050781 | 0.512385 | 0.053892 |

## Selector Aggregate

```json
{
  "num_variants": 5,
  "num_successful": 5,
  "failed_variants": [],
  "best_by_selected_teacher_importance": {
    "variant_name": "weighted_mse_alpha2",
    "variant_run_name": "weighted_mse_alpha2_seed0",
    "loss_type": "weighted_mse",
    "seed": 0,
    "success": true,
    "num_steps": 300,
    "initial_loss": 0.09764230996370316,
    "final_loss": 0.043495502322912216,
    "best_loss": 0.016780130565166473,
    "loss_decreased": true,
    "test_importance_mse": 0.0461021289229393,
    "test_importance_mae": 0.17089954018592834,
    "test_pearson_corr_mean": -0.011240443214774132,
    "test_target_top1_overlap": 0.0,
    "test_target_topk_overlap": 0.05078125,
    "test_selected_teacher_importance_mean": 0.512384831905365,
    "test_random_teacher_importance_mean": 0.4584925174713135,
    "test_selected_vs_random_importance_gap": 0.053892314434051514,
    "score_mean": 0.5207911133766174,
    "score_std": 0.10554179549217224,
    "target_mean": 0.4699629247188568,
    "target_std": 0.18634603917598724,
    "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/checkpoints/student_selector_step_000300.pt",
    "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/metrics.jsonl",
    "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/summary.json",
    "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/eval_summary.json",
    "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0",
    "device": "cuda",
    "topk": 16,
    "train_num_samples": 100,
    "test_num_samples": 16,
    "num_tokens": 392,
    "token_dim": 768,
    "loss_config": {
      "name": "weighted_mse_alpha2",
      "type": "weighted_mse",
      "seed": 0,
      "alpha": 2.0,
      "topk": 16
    },
    "train_eval_metrics": {
      "importance_mse": 0.017144901677966118,
      "importance_mae": 0.1000242531299591,
      "pearson_corr_mean": 0.6580300331115723,
      "target_top1_overlap": 0.07999999821186066,
      "target_topk_overlap": 0.295625,
      "selected_teacher_importance_mean": 0.7368444800376892,
      "random_teacher_importance_mean": 0.5122919678688049,
      "selected_vs_random_importance_gap": 0.22455251216888428,
      "score_mean": 0.5034849047660828,
      "score_std": 0.14014092087745667,
      "score_min": 0.06424645334482193,
      "score_max": 0.9426285624504089,
      "target_mean": 0.514139711856842,
      "target_std": 0.19880874454975128,
      "target_min": 0.0,
      "target_max": 1.0,
      "num_samples": 100,
      "num_tokens": 392,
      "topk": 16
    },
    "test_eval_metrics": {
      "importance_mse": 0.0461021289229393,
      "importance_mae": 0.17089954018592834,
      "pearson_corr_mean": -0.011240443214774132,
      "target_top1_overlap": 0.0,
      "target_topk_overlap": 0.05078125,
      "selected_teacher_importance_mean": 0.512384831905365,
      "random_teacher_importance_mean": 0.4584925174713135,
      "selected_vs_random_importance_gap": 0.053892314434051514,
      "score_mean": 0.5207911133766174,
      "score_std": 0.10554179549217224,
      "score_min": 0.19795392453670502,
      "score_max": 0.9451943039894104,
      "target_mean": 0.4699629247188568,
      "target_std": 0.18634603917598724,
      "target_min": 0.0,
      "target_max": 1.0,
      "num_samples": 16,
      "num_tokens": 392,
      "topk": 16
    }
  },
  "best_by_target_topk_overlap": {
    "variant_name": "topk_bce",
    "variant_run_name": "topk_bce_seed0",
    "loss_type": "topk_bce",
    "seed": 0,
    "success": true,
    "num_steps": 300,
    "initial_loss": 1.3922330141067505,
    "final_loss": 0.5498328804969788,
    "best_loss": 0.3215303122997284,
    "loss_decreased": true,
    "test_importance_mse": 0.1872028112411499,
    "test_importance_mae": 0.388860821723938,
    "test_pearson_corr_mean": 0.01380588673055172,
    "test_target_top1_overlap": 0.0,
    "test_target_topk_overlap": 0.18359375,
    "test_selected_teacher_importance_mean": 0.4364883303642273,
    "test_random_teacher_importance_mean": 0.4584925174713135,
    "test_selected_vs_random_importance_gap": -0.02200418710708618,
    "score_mean": 0.16488496959209442,
    "score_std": 0.2521185576915741,
    "target_mean": 0.4699629247188568,
    "target_std": 0.18634603917598724,
    "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/topk_bce_seed0/checkpoints/student_selector_step_000300.pt",
    "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/topk_bce_seed0/metrics.jsonl",
    "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/topk_bce_seed0/summary.json",
    "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/topk_bce_seed0/eval_summary.json",
    "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/topk_bce_seed0",
    "device": "cuda",
    "topk": 16,
    "train_num_samples": 100,
    "test_num_samples": 16,
    "num_tokens": 392,
    "token_dim": 768,
    "loss_config": {
      "name": "topk_bce",
      "type": "topk_bce",
      "seed": 0,
      "bce_loss_weight": 1.0,
      "topk": 16
    },
    "train_eval_metrics": {
      "importance_mse": 0.2014135718345642,
      "importance_mae": 0.4039130210876465,
      "pearson_corr_mean": 0.33517196774482727,
      "target_top1_overlap": 0.15000000596046448,
      "target_topk_overlap": 0.543125,
      "selected_teacher_importance_mean": 0.6879673600196838,
      "random_teacher_importance_mean": 0.5122919678688049,
      "selected_vs_random_importance_gap": 0.1756753921508789,
      "score_mean": 0.1548847258090973,
      "score_std": 0.25551918148994446,
      "score_min": 8.26540524911934e-09,
      "score_max": 0.9999405145645142,
      "target_mean": 0.514139711856842,
      "target_std": 0.19880874454975128,
      "target_min": 0.0,
      "target_max": 1.0,
      "num_samples": 100,
      "num_tokens": 392,
      "topk": 16
    },
    "test_eval_metrics": {
      "importance_mse": 0.1872028112411499,
      "importance_mae": 0.388860821723938,
      "pearson_corr_mean": 0.01380588673055172,
      "target_top1_overlap": 0.0,
      "target_topk_overlap": 0.18359375,
      "selected_teacher_importance_mean": 0.4364883303642273,
      "random_teacher_importance_mean": 0.4584925174713135,
      "selected_vs_random_importance_gap": -0.02200418710708618,
      "score_mean": 0.16488496959209442,
      "score_std": 0.2521185576915741,
      "score_min": 5.1745423945703806e-08,
      "score_max": 0.9990449547767639,
      "target_mean": 0.4699629247188568,
      "target_std": 0.18634603917598724,
      "target_min": 0.0,
      "target_max": 1.0,
      "num_samples": 16,
      "num_tokens": 392,
      "topk": 16
    }
  },
  "best_by_importance_mse": {
    "variant_name": "weighted_mse_alpha2",
    "variant_run_name": "weighted_mse_alpha2_seed0",
    "loss_type": "weighted_mse",
    "seed": 0,
    "success": true,
    "num_steps": 300,
    "initial_loss": 0.09764230996370316,
    "final_loss": 0.043495502322912216,
    "best_loss": 0.016780130565166473,
    "loss_decreased": true,
    "test_importance_mse": 0.0461021289229393,
    "test_importance_mae": 0.17089954018592834,
    "test_pearson_corr_mean": -0.011240443214774132,
    "test_target_top1_overlap": 0.0,
    "test_target_topk_overlap": 0.05078125,
    "test_selected_teacher_importance_mean": 0.512384831905365,
    "test_random_teacher_importance_mean": 0.4584925174713135,
    "test_selected_vs_random_importance_gap": 0.053892314434051514,
    "score_mean": 0.5207911133766174,
    "score_std": 0.10554179549217224,
    "target_mean": 0.4699629247188568,
    "target_std": 0.18634603917598724,
    "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/checkpoints/student_selector_step_000300.pt",
    "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/metrics.jsonl",
    "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/summary.json",
    "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/eval_summary.json",
    "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0",
    "device": "cuda",
    "topk": 16,
    "train_num_samples": 100,
    "test_num_samples": 16,
    "num_tokens": 392,
    "token_dim": 768,
    "loss_config": {
      "name": "weighted_mse_alpha2",
      "type": "weighted_mse",
      "seed": 0,
      "alpha": 2.0,
      "topk": 16
    },
    "train_eval_metrics": {
      "importance_mse": 0.017144901677966118,
      "importance_mae": 0.1000242531299591,
      "pearson_corr_mean": 0.6580300331115723,
      "target_top1_overlap": 0.07999999821186066,
      "target_topk_overlap": 0.295625,
      "selected_teacher_importance_mean": 0.7368444800376892,
      "random_teacher_importance_mean": 0.5122919678688049,
      "selected_vs_random_importance_gap": 0.22455251216888428,
      "score_mean": 0.5034849047660828,
      "score_std": 0.14014092087745667,
      "score_min": 0.06424645334482193,
      "score_max": 0.9426285624504089,
      "target_mean": 0.514139711856842,
      "target_std": 0.19880874454975128,
      "target_min": 0.0,
      "target_max": 1.0,
      "num_samples": 100,
      "num_tokens": 392,
      "topk": 16
    },
    "test_eval_metrics": {
      "importance_mse": 0.0461021289229393,
      "importance_mae": 0.17089954018592834,
      "pearson_corr_mean": -0.011240443214774132,
      "target_top1_overlap": 0.0,
      "target_topk_overlap": 0.05078125,
      "selected_teacher_importance_mean": 0.512384831905365,
      "random_teacher_importance_mean": 0.4584925174713135,
      "selected_vs_random_importance_gap": 0.053892314434051514,
      "score_mean": 0.5207911133766174,
      "score_std": 0.10554179549217224,
      "score_min": 0.19795392453670502,
      "score_max": 0.9451943039894104,
      "target_mean": 0.4699629247188568,
      "target_std": 0.18634603917598724,
      "target_min": 0.0,
      "target_max": 1.0,
      "num_samples": 16,
      "num_tokens": 392,
      "topk": 16
    }
  },
  "mean_selected_teacher_importance": 0.46760536432266236,
  "mean_target_topk_overlap": 0.103125,
  "sanity_gate": {
    "checks": {
      "selector_metrics_finite": true,
      "mse_only_variant_success": true,
      "non_mse_only_variant_success": true,
      "best_selector_importance_finite": true
    },
    "pass": true
  }
}
```

## Downstream StudentWorldModel Results

| variant | student_future_mse | teacher_mse | student_teacher_ratio | token_retention_ratio | selector_target_topk_overlap | selected_teacher_importance_mean | selected_vs_random_importance_gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid_weighted_mse_rank_bce | 1.83763584 | 1.68688138 | 1.089369 | 0.040816 | 0.144531 | 0.441189 | -0.017304 |
| mse_only | 1.83880289 | 1.68688138 | 1.090061 | 0.040816 | 0.023438 | 0.479386 | 0.020893 |
| mse_rank_w0p1 | 1.94753659 | 1.68688138 | 1.154519 | 0.040816 | 0.113281 | 0.468579 | 0.010086 |
| topk_bce | 1.82720673 | 1.68688138 | 1.083186 | 0.040816 | 0.183594 | 0.436488 | -0.022004 |
| weighted_mse_alpha2 | 1.76074374 | 1.68688138 | 1.043786 | 0.040816 | 0.050781 | 0.512385 | 0.053892 |

## Downstream Aggregate

```json
{
  "num_variants": 5,
  "num_successful": 5,
  "failed_variants": [],
  "best_by_student_future_mse": {
    "policy": "learned_selector",
    "seed": 0,
    "num_steps": 500,
    "initial_loss": 12.982955932617188,
    "final_loss": 1.3612430095672607,
    "best_loss": 0.9276008009910583,
    "loss_decreased": true,
    "student_future_mse": 1.7607437372207642,
    "student_mse": 1.7607437372207642,
    "teacher_future_mse": 1.686881383260091,
    "teacher_mse": 1.686881383260091,
    "student_teacher_gap": 0.07386235396067309,
    "student_teacher_ratio": 1.0437863353604186,
    "token_retention_ratio": 0.04081632653061224,
    "selector_target_top1_overlap": 0.0,
    "selector_target_topk_overlap": 0.05078125,
    "selected_teacher_importance_mean": 0.512384831905365,
    "random_teacher_importance_mean": 0.4584925174713135,
    "selected_vs_random_importance_gap": 0.053892314434051514,
    "selected_top1_hit_rate": null,
    "selected_topk_hit_rate": null,
    "selected_key_coverage": null,
    "selected_key_fraction": null,
    "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/downstream_student_world_models/weighted_mse_alpha2_seed0/checkpoints/student_world_model_step_000500.pt",
    "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/downstream_student_world_models/weighted_mse_alpha2_seed0/metrics.jsonl",
    "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/downstream_student_world_models/weighted_mse_alpha2_seed0/summary.json",
    "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/downstream_student_world_models/weighted_mse_alpha2_seed0/eval_summary.json",
    "gap_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/downstream_student_world_models/weighted_mse_alpha2_seed0/teacher_student_gap_summary.json",
    "device": "cuda",
    "dataset_size": 100,
    "train_num_samples": 100,
    "test_num_samples": 16,
    "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/downstream_student_world_models/weighted_mse_alpha2_seed0",
    "topk": 16,
    "total_tokens": 392,
    "token_dim": 768,
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
    "variant_name": "weighted_mse_alpha2",
    "variant_run_name": "weighted_mse_alpha2_seed0",
    "loss_type": "weighted_mse",
    "selector_checkpoint": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/checkpoints/student_selector_step_000300.pt",
    "selector_checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/selector_ablation_bair_videomae_smoke_v1/selectors/weighted_mse_alpha2_seed0/checkpoints/student_selector_step_000300.pt",
    "success": true
  },
  "step12_reference": {
    "summary_json": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/baseline_summary.json",
    "random_k_mean": {
      "method": "step12_random_k_mean",
      "student_future_mse": 1.8781254159079657,
      "selected_teacher_importance_mean": 0.4708937505880992,
      "selector_target_topk_overlap": 0.036458333333333336
    },
    "uniform_k": {
      "policy": "uniform_k",
      "seed": 0,
      "num_steps": 500,
      "initial_loss": 13.027299880981445,
      "final_loss": 1.365092158317566,
      "best_loss": 0.9896382093429565,
      "loss_decreased": true,
      "student_future_mse": 1.8350162506103516,
      "student_mse": 1.8350162506103516,
      "teacher_future_mse": 1.686881383260091,
      "teacher_mse": 1.686881383260091,
      "student_teacher_gap": 0.1481348673502605,
      "student_teacher_ratio": 1.0878158172947365,
      "token_retention_ratio": 0.04081632653061224,
      "selector_target_top1_overlap": 0.0,
      "selector_target_topk_overlap": 0.0546875,
      "selected_teacher_importance_mean": 0.48264437913894653,
      "random_teacher_importance_mean": 0.4584925174713135,
      "selected_vs_random_importance_gap": 0.024151861667633057,
      "selected_top1_hit_rate": null,
      "selected_topk_hit_rate": null,
      "selected_key_coverage": null,
      "selected_key_fraction": null,
      "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/uniform_k_seed0/checkpoints/student_world_model_step_000500.pt",
      "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/uniform_k_seed0/metrics.jsonl",
      "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/uniform_k_seed0/summary.json",
      "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/uniform_k_seed0/eval_summary.json",
      "gap_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/uniform_k_seed0/teacher_student_gap_summary.json",
      "device": "cuda",
      "dataset_size": 100,
      "train_num_samples": 100,
      "test_num_samples": 16,
      "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/uniform_k_seed0",
      "topk": 16,
      "total_tokens": 392,
      "token_dim": 768,
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
      }
    },
    "teacher_importance_topk": {
      "policy": "teacher_importance_topk",
      "seed": 0,
      "num_steps": 500,
      "initial_loss": 13.014167785644531,
      "final_loss": 1.3775962591171265,
      "best_loss": 1.0255603790283203,
      "loss_decreased": true,
      "student_future_mse": 1.760870138804118,
      "student_mse": 1.760870138804118,
      "teacher_future_mse": 1.686881383260091,
      "teacher_mse": 1.686881383260091,
      "student_teacher_gap": 0.07398875554402684,
      "student_teacher_ratio": 1.0438612674715961,
      "token_retention_ratio": 0.04081632653061224,
      "selector_target_top1_overlap": 1.0,
      "selector_target_topk_overlap": 0.99609375,
      "selected_teacher_importance_mean": 0.835728645324707,
      "random_teacher_importance_mean": 0.4584925174713135,
      "selected_vs_random_importance_gap": 0.37723612785339355,
      "selected_top1_hit_rate": null,
      "selected_topk_hit_rate": null,
      "selected_key_coverage": null,
      "selected_key_fraction": null,
      "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/teacher_importance_topk_seed0/checkpoints/student_world_model_step_000500.pt",
      "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/teacher_importance_topk_seed0/metrics.jsonl",
      "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/teacher_importance_topk_seed0/summary.json",
      "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/teacher_importance_topk_seed0/eval_summary.json",
      "gap_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/teacher_importance_topk_seed0/teacher_student_gap_summary.json",
      "device": "cuda",
      "dataset_size": 100,
      "train_num_samples": 100,
      "test_num_samples": 16,
      "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/teacher_importance_topk_seed0",
      "topk": 16,
      "total_tokens": 392,
      "token_dim": 768,
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
      }
    },
    "learned_selector": {
      "policy": "learned_selector",
      "seed": 0,
      "num_steps": 500,
      "initial_loss": 12.903804779052734,
      "final_loss": 1.3812907934188843,
      "best_loss": 1.0424336194992065,
      "loss_decreased": true,
      "student_future_mse": 1.8413029511769612,
      "student_mse": 1.8413029511769612,
      "teacher_future_mse": 1.686881383260091,
      "teacher_mse": 1.686881383260091,
      "student_teacher_gap": 0.15442156791687012,
      "student_teacher_ratio": 1.0915426356881317,
      "token_retention_ratio": 0.04081632653061224,
      "selector_target_top1_overlap": 0.25,
      "selector_target_topk_overlap": 0.1640625,
      "selected_teacher_importance_mean": 0.5927466750144958,
      "random_teacher_importance_mean": 0.4584925174713135,
      "selected_vs_random_importance_gap": 0.13425415754318237,
      "selected_top1_hit_rate": null,
      "selected_topk_hit_rate": null,
      "selected_key_coverage": null,
      "selected_key_fraction": null,
      "checkpoint_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/learned_selector_seed0/checkpoints/student_world_model_step_000500.pt",
      "metrics_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/learned_selector_seed0/metrics.jsonl",
      "summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/learned_selector_seed0/summary.json",
      "eval_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/learned_selector_seed0/eval_summary.json",
      "gap_summary_path": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/learned_selector_seed0/teacher_student_gap_summary.json",
      "device": "cuda",
      "dataset_size": 100,
      "train_num_samples": 100,
      "test_num_samples": 16,
      "run_dir": "/home/ubuntu22/tgpawb_world_model/runs/baseline_comparison_bair_videomae_smoke_v1/learned_selector_seed0",
      "topk": 16,
      "total_tokens": 392,
      "token_dim": 768,
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
      }
    }
  },
  "comparison": {
    "downstream_improvement_over_step12_learned": true,
    "best_downstream_vs_step12_learned_mse_delta": -0.08055921395619703,
    "any_variant_selected_importance_gt_step12_learned": false,
    "any_variant_topk_overlap_gt_step12_learned": true,
    "best_downstream_beats_step12_uniform_mse": true,
    "best_downstream_vs_step12_uniform_mse_delta": -0.0742725133895874,
    "best_downstream_gap_to_teacher_importance_topk_mse": -0.0001264015833537524
  },
  "sanity_gate": {
    "checks": {
      "downstream_result_present": true,
      "downstream_metrics_finite": true,
      "best_downstream_mse_finite": true
    },
    "pass": true
  }
}
```

## Comparison With Step 12

| method | student_future_mse | selected_teacher_importance_mean | topk_overlap | note |
| --- | ---: | ---: | ---: | --- |
| step12_random_k_mean | 1.87812542 | 0.470894 | 0.036458 | mean over seeds 0,1,2 |
| step12_uniform_k | 1.83501625 | 0.482644 | 0.054688 | seed0 |
| step12_teacher_importance_topk | 1.76087014 | 0.835729 | 0.996094 | oracle-like upper bound seed0 |
| step12_learned_selector | 1.84130295 | 0.592747 | 0.164062 | Step 12 learned selector seed0 |

## Combined Sanity Gate

```json
{
  "selector_pass": true,
  "downstream_pass": true,
  "pass": true
}
```


## Best Variants

- best selector by selected importance: `weighted_mse_alpha2`
- best selector by topK overlap: `topk_bce`
- best downstream by MSE: `weighted_mse_alpha2`

## Improvement Flags

```json
{
  "downstream_improvement_over_step12_learned": true,
  "best_downstream_vs_step12_learned_mse_delta": -0.08055921395619703,
  "any_variant_selected_importance_gt_step12_learned": false,
  "any_variant_topk_overlap_gt_step12_learned": true,
  "best_downstream_beats_step12_uniform_mse": true,
  "best_downstream_vs_step12_uniform_mse_delta": -0.0742725133895874,
  "best_downstream_gap_to_teacher_importance_topk_mse": -0.0001264015833537524
}
```

## Resource Usage

```json
{
  "before": {
    "ram_used_gib": 6.2236785888671875,
    "ram_total_gib": 30.40261459350586,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.9208984375,
    "gpu_mem_used_gib": 0.7470703125
  },
  "after": {
    "ram_used_gib": 8.619491577148438,
    "ram_total_gib": 30.40261459350586,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_mem_total_gib": 15.9208984375,
    "gpu_mem_used_gib": 1.13671875
  },
  "elapsed_time_sec": 12.449056148529053,
  "max_ram_used_gib": 8.619491577148438,
  "max_gpu_mem_used_gib": 1.13671875,
  "oom": false,
  "gpu_mem_fraction": 0.07139790222658407,
  "cloud_recommendation": "not required for Step 13 smoke"
}
```

## Checks

```json
{
  "selector_summary_json_exists": true,
  "selector_summary_csv_exists": true,
  "selector_summary_md_exists": true,
  "downstream_summary_json_exists": true,
  "downstream_summary_csv_exists": true,
  "downstream_summary_md_exists": true,
  "combined_report_json_exists": true,
  "combined_report_md_exists": true,
  "mse_only_variant_success": true,
  "non_mse_only_variant_success": true,
  "downstream_result_finite": true,
  "selector_sanity_gate_pass": true,
  "downstream_sanity_gate_pass": true,
  "combined_sanity_gate_pass": true,
  "ram_below_warning": true,
  "oom_false": true
}
```

BAIR_SELECTOR_ABLATION_SMOKE_PASS = true
