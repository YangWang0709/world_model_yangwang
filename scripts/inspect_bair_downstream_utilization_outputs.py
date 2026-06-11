"""Inspect Step16 downstream utilization outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.8f}"
    except (TypeError, ValueError):
        return str(value)


def inspect_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    summary_path = root / "downstream_utilization_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = {
        "run_dir": str(root),
        "phase_a_summary": str(root / "phase_a_summary.json"),
        "phase_b_summary": str(root / "phase_b_summary.json"),
        "summary": str(summary_path),
        "best_variant": summary.get("phase_b_best_mse_variant"),
        "best_mse_mean": summary.get("phase_b_best_mse_mean"),
        "sanity_gate_pass": summary.get("sanity_gate_pass"),
        "cloud_required": summary.get("cloud_required"),
    }
    print("STEP16_DOWNSTREAM_UTILIZATION_OUTPUTS")
    for key, value in payload.items():
        print(f"{key}: {value}")
    print("PHASE_B_VARIANTS")
    for row in summary.get("phase_b_variants", []):
        print(
            "{variant}: mse={mse} std={std} selected_importance={importance}".format(
                variant=row.get("variant"),
                mse=_fmt(row.get("student_future_mse_mean")),
                std=_fmt(row.get("student_future_mse_std")),
                importance=_fmt(row.get("selected_teacher_importance_mean")),
            )
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="/home/ubuntu22/tgpawb_world_model/runs/downstream_utilization_bair_1000_128_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_run(args.run_dir)


if __name__ == "__main__":
    main()
