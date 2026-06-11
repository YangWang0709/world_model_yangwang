# BAIR Dataset Schema

Step 11A exports BAIR Robot Pushing small into the project-local `.pt` clip format. The exported data is intentionally ignored by git.

## Directory Layout

```text
data/bair_robot_pushing_small_subset/
  train/
    metadata.jsonl
    clips/
      bair_train_000000.pt
  test/
    metadata.jsonl
    clips/
      bair_test_000000.pt
```

## Clip Payload

Each clip is a `torch.save` payload:

```python
{
    "video": torch.Tensor,              # [T, C, H, W], float32, [0, 1]
    "task_text": "predict robot pushing future visual dynamics",
    "sample_id": str,
    "fps": 5,
    "source": "bair_robot_pushing_small",
    "split": "train" or "test",
    "camera": "image_main",
    "actions": Optional[torch.Tensor],
    "endeffector_pos": Optional[torch.Tensor],
    "metadata": dict,
}
```

Step 11A uses `total_frames=8`, `past_len=4`, `future_len=4`, and resizes the 64x64 BAIR frames to 224x224.

## Metadata Row

Each `metadata.jsonl` row is a JSON object:

```json
{
  "sample_id": "bair_train_000000",
  "source_type": "pt_clip",
  "clip_path": "clips/bair_train_000000.pt",
  "path": "clips/bair_train_000000.pt",
  "task_text": "predict robot pushing future visual dynamics",
  "num_frames": 8,
  "fps": 5,
  "height": 224,
  "width": 224,
  "source": "bair_robot_pushing_small",
  "split": "train",
  "camera": "image_main",
  "has_action": true,
  "has_endeffector_pos": true
}
```

## Loader Output

`BAIRRobotPushingDataset` reads only exported `.pt` clips and does not import TensorFlow.

```python
{
    "past_video": Tensor,       # [4, 3, 224, 224]
    "future_video": Tensor,     # [4, 3, 224, 224]
    "task_text": str,
    "sample_id": str,
    "actions": Optional[Tensor],
    "endeffector_pos": Optional[Tensor],
    "metadata": dict,
}
```

## Git Boundary

The TFDS cache, exported `.pt` clips, token shards, checkpoints, logs, and run directories are ignored and must not be committed.
