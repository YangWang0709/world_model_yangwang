# BAIR VideoMAE Token Extraction Smoke Report

- command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_bair_videomae_token_extraction.py`
- input subset dir: `/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset`
- VideoMAE model path: `/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local`
- train samples: `100`
- test samples: `16`
- train output dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train`
- test output dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test`
- train shard count: `25`
- test shard count: `4`
- first train shard shape: `[4, 392, 768]`
- first test shard shape: `[4, 392, 768]`
- elapsed_time_sec: `4.323`
- oom: `False`
- cloud_recommendation: `not_needed_for_step11b`
- BAIR_VIDEOMAE_TOKEN_EXTRACTION_SMOKE_PASS: `true`

## Resource Before

```json
{
  "disk_free_gib": 189.052,
  "gpu": {
    "nvidia_smi": [
      {
        "memory_free_mib": 15018,
        "memory_total_mib": 16303,
        "memory_used_mib": 815,
        "name": "NVIDIA GeForce RTX 5080"
      }
    ],
    "torch_cuda_allocated_mib": 0.0,
    "torch_cuda_available": true,
    "torch_cuda_reserved_mib": 0.0
  },
  "ram_available_gib": 24.793,
  "ram_total_gib": 30.403,
  "ram_used_gib": 5.61
}
```

## Resource After

```json
{
  "disk_free_gib": 188.791,
  "gpu": {
    "nvidia_smi": [
      {
        "memory_free_mib": 14290,
        "memory_total_mib": 16303,
        "memory_used_mib": 1542,
        "name": "NVIDIA GeForce RTX 5080"
      }
    ],
    "torch_cuda_allocated_mib": 8.125,
    "torch_cuda_available": true,
    "torch_cuda_reserved_mib": 402.0
  },
  "ram_available_gib": 24.134,
  "ram_total_gib": 30.403,
  "ram_used_gib": 6.269
}
```

## First Train Shard

```json
{
  "encoder_config": {
    "actual_encoder": "videomae",
    "model_name_or_path": "/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local",
    "num_tokens": 392,
    "output_mode": "last_hidden_state",
    "requested_encoder": "videomae",
    "temporal_pad_to": null,
    "token_dim": 768,
    "used_fallback": false
  },
  "encoder_name": "videomae",
  "first_metadata": {
    "action_shape": [
      8,
      4
    ],
    "camera": "image_main",
    "clip_path": "clips/bair_train_000000.pt",
    "endeffector_pos_shape": [
      8,
      3
    ],
    "fps": 5,
    "has_action": true,
    "has_endeffector_pos": true,
    "height": 224,
    "num_frames": 8,
    "original_num_frames": 30,
    "path": "clips/bair_train_000000.pt",
    "required_frames": 8,
    "resolved_path": "/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset/train/clips/bair_train_000000.pt",
    "sample_id": "bair_train_000000",
    "source": "bair_robot_pushing_small",
    "source_type": "pt_clip",
    "split": "train",
    "task_text": "predict robot pushing future visual dynamics",
    "width": 224
  },
  "first_sample_id": "bair_train_000000",
  "future_tokens_rank": 3,
  "future_tokens_shape": [
    4,
    392,
    768
  ],
  "num_samples": 4,
  "past_tokens_shape": [
    4,
    392,
    768
  ],
  "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000000.pt",
  "schema_version": "0.1.0",
  "split": "train"
}
```

## First Test Shard

```json
{
  "encoder_config": {
    "actual_encoder": "videomae",
    "model_name_or_path": "/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local",
    "num_tokens": 392,
    "output_mode": "last_hidden_state",
    "requested_encoder": "videomae",
    "temporal_pad_to": null,
    "token_dim": 768,
    "used_fallback": false
  },
  "encoder_name": "videomae",
  "first_metadata": {
    "action_shape": [
      8,
      4
    ],
    "camera": "image_main",
    "clip_path": "clips/bair_test_000000.pt",
    "endeffector_pos_shape": [
      8,
      3
    ],
    "fps": 5,
    "has_action": true,
    "has_endeffector_pos": true,
    "height": 224,
    "num_frames": 8,
    "original_num_frames": 30,
    "path": "clips/bair_test_000000.pt",
    "required_frames": 8,
    "resolved_path": "/home/ubuntu22/tgpawb_world_model/data/bair_robot_pushing_small_subset/test/clips/bair_test_000000.pt",
    "sample_id": "bair_test_000000",
    "source": "bair_robot_pushing_small",
    "source_type": "pt_clip",
    "split": "test",
    "task_text": "predict robot pushing future visual dynamics",
    "width": 224
  },
  "first_sample_id": "bair_test_000000",
  "future_tokens_rank": 3,
  "future_tokens_shape": [
    4,
    392,
    768
  ],
  "num_samples": 4,
  "past_tokens_shape": [
    4,
    392,
    768
  ],
  "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test/tokens_shard_000000.pt",
  "schema_version": "0.1.0",
  "split": "test"
}
```

## Extraction Summary

```json
{
  "actual_encoder": "videomae",
  "config": "/home/ubuntu22/tgpawb_world_model/configs/token_extraction_bair_videomae_smoke.yaml",
  "dataset_name": "bair_robot_pushing_small_subset",
  "elapsed_time_sec": 4.258,
  "encoder_availability": {
    "allow_download": false,
    "available": true,
    "cache_dir": "/home/ubuntu22/tgpawb_world_model/model_cache/huggingface",
    "device": "cuda",
    "effective_num_frames": 4,
    "ignore_mismatched_sizes": true,
    "image_size": 224,
    "local_files_only": true,
    "model_name_or_path": "/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local",
    "output_mode": "last_hidden_state",
    "pretrained_num_frames": 16,
    "reason": "VideoMAE model loaded"
  },
  "fallback_reason": null,
  "output_root": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke",
  "requested_encoder": "videomae",
  "split_summaries": {
    "test": {
      "actual_encoder": "videomae",
      "batch_size": 1,
      "cache_dir": "/home/ubuntu22/tgpawb_world_model/model_cache/huggingface",
      "config": "/home/ubuntu22/tgpawb_world_model/configs/token_extraction_bair_videomae_smoke.yaml",
      "dataset_size": 16,
      "device": "cuda",
      "elapsed_time_sec": 4.258,
      "encoder_availability": {
        "allow_download": false,
        "available": true,
        "cache_dir": "/home/ubuntu22/tgpawb_world_model/model_cache/huggingface",
        "device": "cuda",
        "effective_num_frames": 4,
        "ignore_mismatched_sizes": true,
        "image_size": 224,
        "local_files_only": true,
        "model_name_or_path": "/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local",
        "output_mode": "last_hidden_state",
        "pretrained_num_frames": 16,
        "reason": "VideoMAE model loaded"
      },
      "encoder_name": "videomae",
      "encoder_runtime_summary": {
        "device": "cuda",
        "effective_input_shape": [
          1,
          4,
          3,
          224,
          224
        ],
        "input_shape": [
          1,
          4,
          3,
          224,
          224
        ],
        "output_mode": "last_hidden_state",
        "output_shape": [
          1,
          392,
          768
        ]
      },
      "fallback_reason": null,
      "max_samples": 16,
      "model_name_or_path": "/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local",
      "num_shards": 4,
      "output_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test",
      "output_token_shape": [
        4,
        392,
        768
      ],
      "requested_encoder": "videomae",
      "shard_size": 4,
      "shards": [
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test/tokens_shard_000000.pt",
          "schema_version": "0.1.0",
          "split": "test"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test/tokens_shard_000001.pt",
          "schema_version": "0.1.0",
          "split": "test"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test/tokens_shard_000002.pt",
          "schema_version": "0.1.0",
          "split": "test"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/test/tokens_shard_000003.pt",
          "schema_version": "0.1.0",
          "split": "test"
        }
      ],
      "split": "test",
      "used_fallback": false
    },
    "train": {
      "actual_encoder": "videomae",
      "batch_size": 1,
      "cache_dir": "/home/ubuntu22/tgpawb_world_model/model_cache/huggingface",
      "config": "/home/ubuntu22/tgpawb_world_model/configs/token_extraction_bair_videomae_smoke.yaml",
      "dataset_size": 100,
      "device": "cuda",
      "elapsed_time_sec": 3.884,
      "encoder_availability": {
        "allow_download": false,
        "available": true,
        "cache_dir": "/home/ubuntu22/tgpawb_world_model/model_cache/huggingface",
        "device": "cuda",
        "effective_num_frames": 4,
        "ignore_mismatched_sizes": true,
        "image_size": 224,
        "local_files_only": true,
        "model_name_or_path": "/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local",
        "output_mode": "last_hidden_state",
        "pretrained_num_frames": 16,
        "reason": "VideoMAE model loaded"
      },
      "encoder_name": "videomae",
      "encoder_runtime_summary": {
        "device": "cuda",
        "effective_input_shape": [
          1,
          4,
          3,
          224,
          224
        ],
        "input_shape": [
          1,
          4,
          3,
          224,
          224
        ],
        "output_mode": "last_hidden_state",
        "output_shape": [
          1,
          392,
          768
        ]
      },
      "fallback_reason": null,
      "max_samples": 100,
      "model_name_or_path": "/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local",
      "num_shards": 25,
      "output_dir": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train",
      "output_token_shape": [
        4,
        392,
        768
      ],
      "requested_encoder": "videomae",
      "shard_size": 4,
      "shards": [
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000000.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000001.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000002.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000003.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000004.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000005.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000006.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000007.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000008.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000009.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000010.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000011.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000012.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000013.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000014.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000015.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000016.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000017.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000018.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000019.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000020.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000021.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000022.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000023.pt",
          "schema_version": "0.1.0",
          "split": "train"
        },
        {
          "encoder_name": "videomae",
          "future_tokens_rank": 3,
          "future_tokens_shape": [
            4,
            392,
            768
          ],
          "num_samples": 4,
          "past_tokens_shape": [
            4,
            392,
            768
          ],
          "path": "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_smoke/train/tokens_shard_000024.pt",
          "schema_version": "0.1.0",
          "split": "train"
        }
      ],
      "split": "train",
      "used_fallback": false
    }
  },
  "splits": [
    "train",
    "test"
  ],
  "total_samples": 116,
  "total_shards": 29,
  "used_fallback": false
}
```

BAIR_VIDEOMAE_TOKEN_EXTRACTION_SMOKE_PASS = true
