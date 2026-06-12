"""BridgeData V2 LongContextSample mapping dry-run with no data download."""

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

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_migration_dryrun.yaml"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "long_context_dataset_feasibility_step19_v1"


def run_bridgedata_dryrun(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_RUN_DIR,
    local_sample_dir: Path | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = LongContextWindowSpec(context_len=16, current_len=4, future_len=4, stride=8)
    window = spec.generate_window_indices(total_frames=48)[0]
    sample = make_placeholder_long_context_sample(
        dataset_name="BridgeData V2",
        trajectory_id="bridgedata_v2_placeholder_traj",
        window=window,
        metadata={
            "source": "fake_metadata_only",
            "actions_saved_as_metadata": True,
            "use_action_as_input": False,
            "language_saved_as_metadata": True,
            "goal_image_saved_as_metadata": True,
        },
    )
    sample["language_instruction"] = "placeholder instruction, not encoded"
    sample["camera_names"] = ["primary"]
    summary = {
        "dataset_name": "BridgeData V2",
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
            "generated_window_count_for_fake_48_frames": len(spec.generate_window_indices(total_frames=48)),
            "first_window": window,
        },
        "placeholder_sample_summary": summarize_long_context_sample(sample),
        "recommended_initial_subset": config["recommended_initial_subset"],
        "estimated_risks": [
            "exact frame counts and file layout need verification on a user-provided tiny subset",
            "full dataset is out of scope for local Step19",
            "goal/language metadata should not be wired into VLM or language encoder in Step20",
        ],
        "recommended_next_step": config["recommended_next_step"],
    }
    (output_dir / "bridgedata_v2_dryrun_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "bridgedata_v2_dryrun_summary.md").write_text(render_bridgedata_markdown(summary), encoding="utf-8")
    return summary


def render_bridgedata_markdown(summary: dict[str, Any]) -> str:
    subset = summary["recommended_initial_subset"]
    lines = [
        "# BridgeData V2 Migration Dry-Run",
        "",
        "- no_download: `true`",
        "- full_download_allowed: `false`",
        "- training_performed: `false`",
        f"- local_sample_dir_exists: `{str(summary['local_sample_dir_exists']).lower()}`",
        "",
        "## Recommended Initial Subset",
        "",
        f"- trajectories: `{subset['trajectories']}`",
        f"- context/current/future: `{subset['context_len']}/{subset['current_len']}/{subset['future_len']}`",
        f"- image_size: `{subset['image_size']}`",
        f"- encoder: `{subset['encoder']}`",
        "- action/language/goal: save as metadata only",
        "",
        "## Why BridgeData V2 First",
        "",
        "- strong fit for goal-image or language-conditioned future work",
        "- smaller and more local-subset-friendly than DROID",
        "- useful next step after BAIR because trajectories and task variation are broader",
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
    summary = run_bridgedata_dryrun(args.config, args.output_dir, args.local_sample_dir)
    print(json.dumps({"dataset_name": summary["dataset_name"], "no_download": summary["no_download"]}, indent=2))


if __name__ == "__main__":
    main()
