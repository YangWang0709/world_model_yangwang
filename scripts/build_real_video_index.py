"""Build metadata.jsonl for Step 9A real_video_minimal inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.real_video_index import build_real_video_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-metadata", required=True)
    parser.add_argument("--source", default="real_video_minimal")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--task-text", default="predict future visual dynamics")
    parser.add_argument("--split", default="real_minimal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_real_video_index(
        input_dir=args.input_dir,
        output_metadata=args.output_metadata,
        source=args.source,
        max_samples=args.max_samples,
        min_frames=args.min_frames,
        task_text=args.task_text,
        split=args.split,
    )
    print(json.dumps(summary, indent=2))
    print(f"REAL_VIDEO_METADATA_WRITTEN = {summary['metadata_path']}")


if __name__ == "__main__":
    main()
