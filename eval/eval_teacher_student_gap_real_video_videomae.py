"""Compare Step 10A Teacher and Step 10D compressed Student on real VideoMAE tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_teacher_student_gap import evaluate_teacher_student_gap
from training.train_student_world_model import load_yaml


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_student_world_model_real_video_videomae_smoke.yaml"
DEFAULT_STUDENT_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "student_world_model_real_video_videomae_smoke_v1"
    / "checkpoints"
    / "student_world_model_step_000500.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--student-checkpoint", default=str(DEFAULT_STUDENT_CHECKPOINT))
    parser.add_argument("--teacher-checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_teacher_student_gap(
        load_yaml(args.config),
        student_checkpoint_path=args.student_checkpoint,
        teacher_checkpoint_path=args.teacher_checkpoint,
    )
    print("TEACHER_STUDENT_GAP_REAL_VIDEOMAE_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
