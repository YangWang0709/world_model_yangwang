# Real Video Resource Report

Command: `python scripts/check_resource_limits.py`

## Summary

- CPU count: `32`
- RAM total GiB: `30.403`
- RAM available GiB: `25.484`
- Disk total GiB: `483.587`
- Disk free GiB: `273.11`
- CUDA available: `True`
- GPU name: `NVIDIA GeForce RTX 5080`
- GPU memory total GiB: `15.461`
- GPU memory free GiB: `14.378`

## Conservative Step 9A Settings

- `batch_size <= 2`
- `num_workers = 0` by default
- `max_samples <= 100`
- `clip_len <= 8`
- `image_size <= 224`

## Warnings

- none

```json
{
  "cpu_count": 32,
  "ram_total_gib": 30.403,
  "ram_available_gib": 25.484,
  "disk_free_gib": 273.11,
  "disk_total_gib": 483.587,
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 5080",
  "gpu_memory_total_gib": 15.461,
  "gpu_memory_free_gib": 14.378,
  "recommended": {
    "batch_size": 2,
    "num_workers": 0,
    "max_samples": 100,
    "clip_len": 8,
    "image_size": 224
  },
  "warnings": []
}
```
