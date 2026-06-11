# Importance Real VideoMAE Smoke Report

Command: `python scripts/smoke_test_predictive_importance_real_video_videomae.py`

- input token shard dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke`
- teacher checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/teacher_real_video_videomae_smoke_v1/checkpoints/teacher_world_model_step_000100.pt`
- output importance dir: `/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke`
- num samples: `8`
- token shape: `[2, 784, 768]`
- token_chunk_size: `16`
- generated shard count: `8`
- GPU memory before: `{'free_gib': 14.507, 'total_gib': 15.461, 'used_gib': 0.954}`
- GPU memory after: `{'free_gib': 14.432, 'total_gib': 15.461, 'used_gib': 1.028}`
- RAM before: `{'total_gib': 30.403, 'available_gib': 25.698, 'used_gib': 4.224}`
- RAM after: `{'total_gib': 30.403, 'available_gib': 25.122, 'used_gib': 4.785}`
- elapsed time sec: `0.185`
- OOM: `False`
- cloud recommendation: `not required for Step 10B; local RTX 5080 smoke succeeded`
- error: `None`

## Checks

```json
{
  "token_shards_exist": true,
  "teacher_checkpoint_exists": true,
  "generation_summary_exists": true,
  "generated_shards_ok": true,
  "importance_scores_shape_ok": true,
  "importance_scores_norm_shape_ok": true,
  "base_losses_shape_ok": true,
  "masked_losses_shape_ok": true,
  "inspect_matches_first": true,
  "eval_samples_ok": true,
  "eval_tokens_ok": true,
  "finite_tensors_ok": true,
  "oom_ok": true,
  "resource_warning_ok": true
}
```

## First Shard Summary

```json
{
  "schema_version": "0.1.0",
  "importance_method": "teacher_token_occlusion",
  "num_samples": 1,
  "num_tokens": 784,
  "importance_scores_shape": [
    1,
    784
  ],
  "importance_scores_norm_shape": [
    1,
    784
  ],
  "base_losses_shape": [
    1
  ],
  "masked_losses_shape": [
    1,
    784
  ],
  "importance_mean": 1.5648880662411102e-06,
  "importance_std": 1.4827156519459095e-05,
  "importance_min": -2.9385089874267578e-05,
  "importance_max": 4.902482032775879e-05,
  "importance_norm_mean": 0.3947202265262604,
  "importance_norm_std": 0.18909798562526703,
  "importance_norm_min": 0.0,
  "importance_norm_max": 1.0,
  "base_loss_mean": 0.1643042266368866,
  "masked_loss_mean": 0.16430579125881195,
  "split": "real_minimal",
  "source_token_shard": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke/tokens_shard_000000.pt"
}
```

## Eval Summary

```json
{
  "importance_dir": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke",
  "num_shards": 8,
  "num_samples": 8,
  "num_tokens": 784,
  "token_dim": 768,
  "shard_files": [
    "importance_shard_000000.pt",
    "importance_shard_000001.pt",
    "importance_shard_000002.pt",
    "importance_shard_000003.pt",
    "importance_shard_000004.pt",
    "importance_shard_000005.pt",
    "importance_shard_000006.pt",
    "importance_shard_000007.pt"
  ],
  "importance_mean": 1.563841465213045e-06,
  "importance_std": 0.00022119932691566646,
  "importance_min": -0.003388732671737671,
  "importance_max": 0.0024299323558807373,
  "normalized_importance_mean": 0.46076127886772156,
  "normalized_importance_std": 0.33224743604660034,
  "normalized_importance_min": 0.0,
  "normalized_importance_max": 1.0,
  "positive_importance_ratio": 0.4832589328289032,
  "base_loss_mean": 0.1661720871925354,
  "base_loss_std": 0.03978302329778671,
  "base_loss_min": 0.10901658236980438,
  "base_loss_max": 0.22318300604820251,
  "masked_loss_mean": 0.16617366671562195,
  "masked_loss_std": 0.03978365659713745,
  "masked_loss_min": 0.10888975113630295,
  "masked_loss_max": 0.22326713800430298,
  "top1_importance_mean": 0.0008337665349245071,
  "top5_importance_mean": 0.0006721671088598669,
  "top10_importance_mean": 0.000566251517739147
}
```

## Generation Summary

```json
{
  "schema_version": "0.1.0",
  "importance_method": "teacher_token_occlusion",
  "teacher_checkpoint": "/home/ubuntu22/tgpawb_world_model/runs/teacher_real_video_videomae_smoke_v1/checkpoints/teacher_world_model_step_000100.pt",
  "source_token_shard_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke",
  "output_dir": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke",
  "device": "cuda",
  "num_source_shards": 4,
  "num_importance_shards": 8,
  "num_samples": 8,
  "num_tokens": 784,
  "token_dim": 768,
  "batch_size": 1,
  "token_chunk_size": 16,
  "max_samples": 8,
  "mask_config": {
    "mask_mode": "zero",
    "mask_value": 0.0,
    "clamp_negative_importance": false,
    "normalize": "minmax_per_sample"
  },
  "importance_shard_files": [
    "importance_shard_000000.pt",
    "importance_shard_000001.pt",
    "importance_shard_000002.pt",
    "importance_shard_000003.pt",
    "importance_shard_000004.pt",
    "importance_shard_000005.pt",
    "importance_shard_000006.pt",
    "importance_shard_000007.pt"
  ],
  "shards": [
    {
      "schema_version": "0.1.0",
      "importance_method": "teacher_token_occlusion",
      "num_samples": 1,
      "num_tokens": 784,
      "importance_scores_shape": [
        1,
        784
      ],
      "importance_scores_norm_shape": [
        1,
        784
      ],
      "base_losses_shape": [
        1
      ],
      "masked_losses_shape": [
        1,
        784
      ],
      "importance_mean": 1.5648880662411102e-06,
      "importance_std": 1.4827156519459095e-05,
      "importance_min": -2.9385089874267578e-05,
      "importance_max": 4.902482032775879e-05,
      "importance_norm_mean": 0.3947202265262604,
      "importance_norm_std": 0.18909798562526703,
      "importance_norm_min": 0.0,
      "importance_norm_max": 1.0,
      "base_loss_mean": 0.1643042266368866,
      "masked_loss_mean": 0.16430579125881195,
      "split": "real_minimal",
      "source_token_shard": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke/tokens_shard_000000.pt",
      "path": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke/importance_shard_000000.pt",
      "source_shard_index": 0,
      "source_batch_range": [
        0,
        1
      ]
    },
    {
      "schema_version": "0.1.0",
      "importance_method": "teacher_token_occlusion",
      "num_samples": 1,
      "num_tokens": 784,
      "importance_scores_shape": [
        1,
        784
      ],
      "importance_scores_norm_shape": [
        1,
        784
      ],
      "base_losses_shape": [
        1
      ],
      "masked_losses_shape": [
        1,
        784
      ],
      "importance_mean": 1.7544406318847905e-06,
      "importance_std": 2.0296927687013522e-05,
      "importance_min": -3.8743019104003906e-05,
      "importance_max": 8.413195610046387e-05,
      "importance_norm_mean": 0.3295826315879822,
      "importance_norm_std": 0.16518357396125793,
      "importance_norm_min": 0.0,
      "importance_norm_max": 1.0,
      "base_loss_mean": 0.22318300604820251,
      "masked_loss_mean": 0.2231847643852234,
      "split": "real_minimal",
      "source_token_shard": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke/tokens_shard_000000.pt",
      "path": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke/importance_shard_000001.pt",
      "source_shard_index": 0,
      "source_batch_range": [
        1,
        2
      ]
    },
    {
      "schema_version": "0.1.0",
      "importance_method": "teacher_token_occlusion",
      "num_samples": 1,
      "num_tokens": 784,
      "importance_scores_shape": [
        1,
        784
      ],
      "importance_scores_norm_shape": [
        1,
        784
      ],
      "base_losses_shape": [
        1
      ],
      "masked_losses_shape": [
        1,
        784
      ],
      "importance_mean": 1.4324311905511422e-06,
      "importance_std": 0.00040020133019424975,
      "importance_min": -0.003388732671737671,
      "importance_max": 0.0006395876407623291,
      "importance_norm_mean": 0.8415828347206116,
      "importance_norm_std": 0.09934695065021515,
      "importance_norm_min": 0.0,
      "importance_norm_max": 1.0,
      "base_loss_mean": 0.15642669796943665,
      "masked_loss_mean": 0.15642812848091125,
      "split": "real_minimal",
      "source_token_shard": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke/tokens_shard_000001.pt",
      "path": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke/importance_shard_000002.pt",
      "source_shard_index": 1,
      "source_batch_range": [
        0,
        1
      ]
    },
    {
      "schema_version": "0.1.0",
      "importance_method": "teacher_token_occlusion",
      "num_samples": 1,
      "num_tokens": 784,
      "importance_scores_shape": [
        1,
        784
      ],
      "importance_scores_norm_shape": [
        1,
        784
      ],
      "base_losses_shape": [
        1
      ],
      "masked_losses_shape": [
        1,
        784
      ],
      "importance_mean": 1.7651603911872371e-06,
      "importance_std": 0.00020776635210495442,
      "importance_min": -0.0015781819820404053,
      "importance_max": 0.0002882331609725952,
      "importance_norm_mean": 0.8465143442153931,
      "importance_norm_std": 0.11131840199232101,
      "importance_norm_min": 0.0,
      "importance_norm_max": 1.0,
      "base_loss_mean": 0.13579052686691284,
      "masked_loss_mean": 0.1357923001050949,
      "split": "real_minimal",
      "source_token_shard": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke/tokens_shard_000001.pt",
      "path": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke/importance_shard_000003.pt",
      "source_shard_index": 1,
      "source_batch_range": [
        1,
        2
      ]
    },
    {
      "schema_version": "0.1.0",
      "importance_method": "teacher_token_occlusion",
      "num_samples": 1,
      "num_tokens": 784,
      "importance_scores_shape": [
        1,
        784
      ],
      "importance_scores_norm_shape": [
        1,
        784
      ],
      "base_losses_shape": [
        1
      ],
      "masked_losses_shape": [
        1,
        784
      ],
      "importance_mean": 1.4993532886364846e-06,
      "importance_std": 0.00019102540682069957,
      "importance_min": -0.0017131567001342773,
      "importance_max": 0.0003322809934616089,
      "importance_norm_mean": 0.838283121585846,
      "importance_norm_std": 0.09339096397161484,
      "importance_norm_min": 0.0,
      "importance_norm_max": 1.0,
      "base_loss_mean": 0.12781661748886108,
      "masked_loss_mean": 0.12781812250614166,
      "split": "real_minimal",
      "source_token_shard": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke/tokens_shard_000002.pt",
      "path": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke/importance_shard_000004.pt",
      "source_shard_index": 2,
      "source_batch_range": [
        0,
        1
      ]
    },
    {
      "schema_version": "0.1.0",
      "importance_method": "teacher_token_occlusion",
      "num_samples": 1,
      "num_tokens": 784,
      "importance_scores_shape": [
        1,
        784
      ],
      "importance_scores_norm_shape": [
        1,
        784
      ],
      "base_losses_shape": [
        1
      ],
      "masked_losses_shape": [
        1,
        784
      ],
      "importance_mean": 1.4751578873983817e-06,
      "importance_std": 0.00021856663806829602,
      "importance_min": -0.00036425888538360596,
      "importance_max": 0.0017672628164291382,
      "importance_norm_mean": 0.17158353328704834,
      "importance_norm_std": 0.10254018753767014,
      "importance_norm_min": 0.0,
      "importance_norm_max": 1.0,
      "base_loss_mean": 0.1920640468597412,
      "masked_loss_mean": 0.1920655071735382,
      "split": "real_minimal",
      "source_token_shard": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke/tokens_shard_000002.pt",
      "path": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke/importance_shard_000005.pt",
      "source_shard_index": 2,
      "source_batch_range": [
        1,
        2
      ]
    },
    {
      "schema_version": "0.1.0",
      "importance_method": "teacher_token_occlusion",
      "num_samples": 1,
      "num_tokens": 784,
      "importance_scores_shape": [
        1,
        784
      ],
      "importance_scores_norm_shape": [
        1,
        784
      ],
      "base_losses_shape": [
        1
      ],
      "masked_losses_shape": [
        1,
        784
      ],
      "importance_mean": 1.4998474853200605e-06,
      "importance_std": 0.00030140389571897686,
      "importance_min": -0.00045228004455566406,
      "importance_max": 0.0024299323558807373,
      "importance_norm_mean": 0.15744151175022125,
      "importance_norm_std": 0.1045738011598587,
      "importance_norm_min": 0.0,
      "importance_norm_max": 1.0,
      "base_loss_mean": 0.22077500820159912,
      "masked_loss_mean": 0.2207764983177185,
      "split": "real_minimal",
      "source_token_shard": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke/tokens_shard_000003.pt",
      "path": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke/importance_shard_000006.pt",
      "source_shard_index": 3,
      "source_batch_range": [
        0,
        1
      ]
    },
    {
      "schema_version": "0.1.0",
      "importance_method": "teacher_token_occlusion",
      "num_samples": 1,
      "num_tokens": 784,
      "importance_scores_shape": [
        1,
        784
      ],
      "importance_scores_norm_shape": [
        1,
        784
      ],
      "base_losses_shape": [
        1
      ],
      "masked_losses_shape": [
        1,
        784
      ],
      "importance_mean": 1.5194527804851532e-06,
      "importance_std": 0.00011120647104689851,
      "importance_min": -0.00012683123350143433,
      "importance_max": 0.0010796785354614258,
      "importance_norm_mean": 0.10638179630041122,
      "importance_norm_std": 0.0921720415353775,
      "importance_norm_min": 0.0,
      "importance_norm_max": 1.0,
      "base_loss_mean": 0.10901658236980438,
      "masked_loss_mean": 0.10901810228824615,
      "split": "real_minimal",
      "source_token_shard": "/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke/tokens_shard_000003.pt",
      "path": "/home/ubuntu22/tgpawb_world_model/data/importance_shards/real_video_videomae_teacher_smoke/importance_shard_000007.pt",
      "source_shard_index": 3,
      "source_batch_range": [
        1,
        2
      ]
    }
  ],
  "importance_mean_across_shards": 1.563841465213045e-06,
  "elapsed_time_sec": 0.181,
  "oom": false,
  "resource_limits": {
    "max_ram_gb_warning": 28,
    "max_gpu_mem_fraction_warning": 0.9,
    "cloud_recommendation_on_oom": true
  }
}
```

IMPORTANCE_REAL_VIDEOMAE_SMOKE_PASS = true
