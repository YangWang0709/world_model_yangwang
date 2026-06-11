# Real Video Dataset Schema

Step 9A defines a minimal real-video interface for validating dataset plumbing before any large frozen video encoder is attached.

## Metadata JSONL

Each line in `metadata.jsonl` is a JSON object:

```json
{
  "sample_id": "real_000000",
  "source_type": "pt_clip",
  "path": "raw/pt_clips/real_000000.pt",
  "task_text": "predict future visual dynamics",
  "num_frames": 8,
  "fps": 5,
  "height": 128,
  "width": 128,
  "source": "real_video_minimal",
  "split": "real_minimal"
}
```

Supported `source_type` values:

- `pt_clip`: stable Step 9A path.
- `frame_folder`: stable path when PIL/Pillow is available.
- `video_file`: optional path, skipped or reported clearly when no lightweight decode backend exists.

Paths may be absolute, but the default index builder writes paths relative to the directory containing `metadata.jsonl`.

## `.pt` Clip Payload

```python
{
    "video": Tensor,      # [T, C, H, W] or [T, H, W, C]
    "task_text": str,
    "sample_id": str,
    "fps": int,
    "source": str,
}
```

The dataset converts clips to float32 `[T, 3, H, W]` with values in `[0, 1]`.

## Dataset Return Schema

`RealVideoClipDataset.__getitem__` returns:

```python
{
    "past_video": Tensor,       # [past_len, 3, image_size, image_size]
    "future_video": Tensor,     # [future_len, 3, image_size, image_size]
    "task_text": str,
    "sample_id": str,
    "metadata": dict,
}
```

Step 9A defaults:

- `past_len = 4`
- `future_len = 4`
- `image_size = 224`
- `split = real_minimal`
- value range `[0, 1]`

## Resource Limits

The dataset reads samples on demand and does not cache the full dataset in RAM. Step 9A keeps conservative defaults:

- `max_samples <= 100`
- `clip_len <= 8`
- `image_size <= 224`
- `batch_size <= 2`
- `num_workers = 0` by default

Generated video clips, real videos, token shards, checkpoints, logs, and run outputs are local-only artifacts and are ignored by git.
