"""DROID LongContextSample mapping dry-run with no data download."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.long_context_dataset_schema import make_placeholder_long_context_sample, summarize_long_context_sample
from data.long_context_window_spec import LongContextWindowSpec

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "droid_migration_dryrun.yaml"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "long_context_dataset_feasibility_step19_v1"


def run_droid_dryrun(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_RUN_DIR,
    local_sample_dir: Path | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = LongContextWindowSpec(context_len=32, current_len=4, future_len=8, stride=16)
    window = spec.generate_window_indices(total_frames=80)[0]
    sample = make_placeholder_long_context_sample(
        dataset_name="DROID",
        trajectory_id="droid_placeholder_episode",
        window=window,
        metadata={
            "source": "fake_metadata_only",
            "actions_saved_as_metadata": True,
            "robot_state_saved_as_metadata": True,
            "use_action_as_input": False,
            "language_saved_as_metadata": True,
            "multicam_metadata_only": True,
        },
    )
    sample["language_instruction"] = "placeholder task label, not encoded"
    sample["camera_names"] = ["exterior_image_1", "exterior_image_2", "wrist_image"]
    summary = {
        "dataset_name": "DROID",
        "stage": config["stage"],
        "no_download": True,
        "full_download_allowed": False,
        "full_download_performed": False,
        "large_download_performed": False,
        "training_performed": False,
        "local_sample_dir": str(local_sample_dir) if local_sample_dir else None,
        "local_sample_dir_exists": bool(local_sample_dir and local_sample_dir.exists()),
        "official_sources": config["official_sources"],
        "known_public_characteristics": config["known_public_characteristics"],
        "proposed_long_context_sample_mapping": config["planned_mapping"],
        "proposed_windows": {
            "context_len": spec.context_len,
            "current_len": spec.current_len,
            "future_len": spec.future_len,
            "generated_window_count_for_fake_80_frames": len(spec.generate_window_indices(total_frames=80)),
            "first_window": window,
        },
        "placeholder_sample_summary": summarize_long_context_sample(sample),
        "recommended_role": config["recommended_role"],
        "estimated_risks": [
            "full dataset storage is not suitable for the current local Step19 scope",
            "multi-camera and raw/RLDS format choices need a separate storage plan",
            "better as a later generalization benchmark than the first local tiny migration",
        ],
        "recommended_next_step": config["recommended_next_step"],
    }
    (output_dir / "droid_dryrun_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "droid_dryrun_summary.md").write_text(render_droid_markdown(summary), encoding="utf-8")
    return summary


def render_droid_markdown(summary: dict[str, Any]) -> str:
    role = summary["recommended_role"]
    lines = [
        "# DROID Migration Dry-Run",
        "",
        "- no_download: `true`",
        "- full_download_allowed: `false`",
        "- training_performed: `false`",
        f"- local_sample_dir_exists: `{str(summary['local_sample_dir_exists']).lower()}`",
        "",
        "## Recommended Role",
        "",
        f"- first_local_migration_target: `{str(role['first_local_migration_target']).lower()}`",
        f"- use_as_future_large_scale_validation: `{str(role['use_as_future_large_scale_validation']).lower()}`",
        f"- require_storage_planning: `{str(role['require_storage_planning']).lower()}`",
        f"- likely_cloud_or_external_disk_for_full_use: `{str(role['likely_cloud_or_external_disk_for_full_use']).lower()}`",
        "",
        "## Why Not First",
        "",
        "- DROID is much broader and heavier than needed for the first local migration.",
        "- It is excellent for later generalization validation once storage and subset tooling are planned.",
        "",
        "## Risks",
        "",
    ]
    lines.extend(f"- {risk}" for risk in summary["estimated_risks"])
    lines.extend(["", f"Recommended next step: {summary['recommended_next_step']}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--local-sample-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = run_droid_dryrun(args.config, args.output_dir, args.local_sample_dir)
    print(json.dumps({"dataset_name": summary["dataset_name"], "no_download": summary["no_download"]}, indent=2))


if __name__ == "__main__":
    main()
