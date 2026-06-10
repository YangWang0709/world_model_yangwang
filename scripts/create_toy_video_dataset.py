"""Generate the tiny Step 3 toy video dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.toy_data import generate_toy_video_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/toy_videos")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--total-frames", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    records = generate_toy_video_dataset(
        output_dir=output_dir,
        num_samples=args.num_samples,
        total_frames=args.total_frames,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"TOY_DATASET_DIR = {output_dir}")
    print(f"TOY_DATASET_SAMPLES = {len(records)}")
    print(f"TOY_DATASET_METADATA = {output_dir / 'metadata.jsonl'}")


if __name__ == "__main__":
    main()

