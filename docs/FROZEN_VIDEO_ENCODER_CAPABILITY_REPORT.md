# Frozen Video Encoder Capability Report

Command: `python scripts/check_video_encoder_capabilities.py`

## Runtime

- Python: `3.11.15`
- torch: `2.7.0+cu128`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 5080`
- GPU memory total/free GiB: `15.461 / 14.44`
- RAM total/available GiB: `30.403 / 25.692`

## Optional Dependencies

- transformers available: `True`
- torchvision available: `True`

## Local Model Cache

- `/home/ubuntu22/.cache/huggingface` exists=`False` entries=`0`
- `/home/ubuntu22/tgpawb_world_model/model_cache` exists=`False` entries=`0`

## Encoder Availability

- VideoMAE: `False`; reason: `model_name_or_path is not configured`
- V-JEPA: `False`; reason: `V-JEPA is a Step 9B placeholder; no weights are downloaded or loaded`

## Download Policy

- `allow_download` defaults to `false`.
- `local_files_only` defaults to `true`.
- This check does not download model weights.

```json
{
  "python": "3.11.15",
  "torch_version": "2.7.0+cu128",
  "cuda_available": true,
  "resource": {
    "cpu_count": 32,
    "ram_total_gib": 30.403,
    "ram_available_gib": 25.692,
    "disk_free_gib": 273.093,
    "disk_total_gib": 483.587,
    "cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_memory_total_gib": 15.461,
    "gpu_memory_free_gib": 14.44,
    "recommended": {
      "batch_size": 2,
      "num_workers": 0,
      "max_samples": 100,
      "clip_len": 8,
      "image_size": 224
    },
    "warnings": []
  },
  "transformers_available": true,
  "torchvision_available": true,
  "huggingface_cache": [
    {
      "path": "/home/ubuntu22/.cache/huggingface",
      "exists": false,
      "num_entries": 0
    },
    {
      "path": "/home/ubuntu22/tgpawb_world_model/model_cache",
      "exists": false,
      "num_entries": 0
    }
  ],
  "videomae": {
    "available": false,
    "reason": "model_name_or_path is not configured",
    "model_name_or_path": null,
    "allow_download": false,
    "local_files_only": true
  },
  "vjepa": {
    "available": false,
    "reason": "V-JEPA is a Step 9B placeholder; no weights are downloaded or loaded",
    "repo_path_exists": false,
    "checkpoint_path_exists": false,
    "model_config_path_exists": false
  },
  "download_policy": {
    "allow_download_default": false,
    "local_files_only_default": true,
    "notes": "Step 9B capability checks never download model weights."
  }
}
```
