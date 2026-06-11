"""Create a tiny local-only .pt clip subset for Step 9A smoke tests."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_motion_clip(sample_index: int, total_frames: int, image_size: int, generator: torch.Generator) -> torch.Tensor:
    """Create a small RGB clip with random texture, moving object, and camera-like noise."""

    y = torch.linspace(0.0, 1.0, image_size).view(1, 1, image_size, 1)
    x = torch.linspace(0.0, 1.0, image_size).view(1, 1, 1, image_size)
    base = torch.cat(
        [
            (0.35 + 0.20 * x).expand(total_frames, 1, image_size, image_size),
            (0.30 + 0.25 * y).expand(total_frames, 1, image_size, image_size),
            (0.25 + 0.15 * (x + y)).expand(total_frames, 1, image_size, image_size),
        ],
        dim=1,
    )
    clip = base + 0.035 * torch.randn((total_frames, 3, image_size, image_size), generator=generator)

    object_size = max(6, image_size // 8)
    phase = (sample_index % 7) / 7.0
    for frame_index in range(total_frames):
        progress = frame_index / max(1, total_frames - 1)
        cx = int((0.12 + 0.72 * progress + 0.06 * math.sin(sample_index)) * image_size)
        cy = int((0.18 + 0.55 * phase + 0.12 * math.sin(progress * math.pi)) * image_size)
        x0 = max(0, min(image_size - object_size, cx - object_size // 2))
        y0 = max(0, min(image_size - object_size, cy - object_size // 2))
        color = torch.tensor(
            [
                0.65 + 0.10 * math.sin(sample_index),
                0.18 + 0.10 * progress,
                0.28 + 0.12 * math.cos(sample_index + frame_index),
            ],
            dtype=torch.float32,
        ).view(3, 1, 1)
        clip[frame_index, :, y0 : y0 + object_size, x0 : x0 + object_size] = color

    camera_jitter = torch.linspace(-0.015, 0.015, total_frames).view(total_frames, 1, 1, 1)
    return (clip + camera_jitter).clamp(0.0, 1.0).contiguous()


def create_real_video_minimal_subset(
    output_dir: str | Path,
    num_samples: int = 16,
    total_frames: int = 8,
    image_size: int = 128,
    seed: int = 42,
) -> dict[str, object]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_samples > 100:
        raise ValueError("Step 9A limits tiny real-video subsets to <= 100 clips")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    clip_paths: list[str] = []
    for sample_index in range(num_samples):
        sample_id = f"real_{sample_index:06d}"
        video = create_motion_clip(sample_index, total_frames, image_size, generator)
        payload = {
            "video": video,
            "task_text": "predict future visual dynamics",
            "sample_id": sample_id,
            "fps": 5,
            "source": "real_video_minimal",
        }
        clip_path = output_path / f"{sample_id}.pt"
        torch.save(payload, clip_path)
        clip_paths.append(str(clip_path))

    summary = {
        "output_dir": str(output_path),
        "num_samples": num_samples,
        "total_frames": total_frames,
        "image_size": image_size,
        "seed": seed,
        "clip_paths": clip_paths,
    }
    (output_path / "subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/real_video_minimal/raw/pt_clips")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--total-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = create_real_video_minimal_subset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        total_frames=args.total_frames,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "clip_paths"}, indent=2))
    print(f"REAL_VIDEO_MINIMAL_SUBSET_WRITTEN = {summary['output_dir']}")


if __name__ == "__main__":
    main()
