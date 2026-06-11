# VideoMAE Download Report

Command: `python scripts/download_videomae_checkpoint.py --model-name <checkpoint-or-repo> --allow-download <true|false>`

- model_name: `/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local`
- cache_dir: `/home/ubuntu22/tgpawb_world_model/model_cache/huggingface`
- source: `local_path`
- allow_download: `False`
- local_files_only: `True`
- download_success: `True`
- elapsed_time_sec: `0.0`
- snapshot_path: `/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local`
- error: `None`

## Disk Usage

- before: `{'total_gib': 483.587, 'used_gib': 190.957, 'free_gib': 272.735, 'cache_size_gib': 0.0}`
- after: `{'total_gib': 483.587, 'used_gib': 190.957, 'free_gib': 272.735, 'cache_size_gib': 0.0}`

## Cached Files Summary

```json
{
  "file_count": 3,
  "total_size_gib": 0.322,
  "shown_files": [
    {
      "path": "config.json",
      "size_mib": 0.022
    },
    {
      "path": "model.safetensors",
      "size_mib": 330.125
    },
    {
      "path": "preprocessor_config.json",
      "size_mib": 0.0
    }
  ],
  "truncated": false
}
```

## Boundary Confirmation

- one VideoMAE checkpoint was allowed for Step 9C
- weights are stored under ignored `model_cache/`
- no model weight files should be committed to git
