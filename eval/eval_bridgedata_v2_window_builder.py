"""Evaluate Step20 BridgeData V2 tiny-window-builder outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_window_builder import load_window_manifest_jsonl

DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "bridgedata_v2_tiny_window_builder_step20_v1"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_window_builder(run_dir: Path = DEFAULT_RUN_DIR, docs_dir: Path = DEFAULT_DOCS_DIR) -> dict[str, Any]:
    inspection = _read_json(run_dir / "bridgedata_v2_subset_inspection.json")
    builder = _read_json(run_dir / "bridgedata_v2_window_builder_summary.json")
    windows = load_window_manifest_jsonl(builder["window_manifest_jsonl"])
    summary = {
        "stage": "bridgedata_v2_tiny_window_builder_step20",
        "pass": bool(builder["sanity_gate_pass"] and len(windows) == builder["num_windows"] and builder["num_windows"] > 0),
        "cloud_required_now": False,
        "full_download_performed": False,
        "large_download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "local_subset_exists": inspection["local_subset_exists"],
        "manifest_found": inspection["manifest_found"],
        "used_fake_manifest": builder["used_fake_manifest"],
        "num_trajectories": builder["num_trajectories"],
        "num_valid_trajectories": builder["num_valid_trajectories"],
        "num_skipped_trajectories": builder["num_skipped_trajectories"],
        "num_windows": builder["num_windows"],
        "window_shape_spec": {
            "context_len": builder["context_len"],
            "current_len": builder["current_len"],
            "future_len": builder["future_len"],
        },
        "metadata_policy": {
            "action_saved_as_metadata": True,
            "action_used_as_input": False,
            "language_saved_as_metadata": True,
            "language_used_as_input": False,
            "goal_image_saved_as_metadata": True,
            "goal_image_used_as_input": False,
        },
        "window_manifest_jsonl": builder["window_manifest_jsonl"],
        "recommended_step21": {
            "name": "BridgeData V2 tiny-subset token extraction dry-run",
            "condition": "only if user provides a real tiny subset or approves tiny download",
            "no_training": True,
        },
    }
    (run_dir / "bridgedata_v2_window_builder_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "bridgedata_v2_window_builder_eval.md").write_text(render_eval_markdown(summary), encoding="utf-8")
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "BRIDGEDATA_V2_TINY_WINDOW_BUILDER_REPORT.md").write_text(
        render_report_markdown(summary), encoding="utf-8"
    )
    (docs_dir / "STEP20_BRIDGEDATA_V2_TINY_WINDOW_BUILDER.md").write_text(render_step_doc(summary), encoding="utf-8")
    (docs_dir / "BRIDGEDATA_V2_USER_SUBSET_FORMAT.md").write_text(render_user_subset_format_doc(), encoding="utf-8")
    return summary


def render_eval_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 Window Builder Eval",
            "",
            f"- pass: `{str(summary['pass']).lower()}`",
            f"- local_subset_exists: `{str(summary['local_subset_exists']).lower()}`",
            f"- used_fake_manifest: `{str(summary['used_fake_manifest']).lower()}`",
            f"- num_windows: `{summary['num_windows']}`",
            f"- context/current/future: `{summary['window_shape_spec']['context_len']}/{summary['window_shape_spec']['current_len']}/{summary['window_shape_spec']['future_len']}`",
            "- no_download: `true`",
            "- no_training: `true`",
            "- no_token_extraction: `true`",
            "",
        ]
    )


def render_report_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 Tiny Window Builder Report",
            "",
            f"- pass: `{str(summary['pass']).lower()}`",
            f"- local subset exists: `{str(summary['local_subset_exists']).lower()}`",
            f"- manifest found: `{str(summary['manifest_found']).lower()}`",
            f"- used fake manifest: `{str(summary['used_fake_manifest']).lower()}`",
            f"- num trajectories: `{summary['num_trajectories']}`",
            f"- valid trajectories: `{summary['num_valid_trajectories']}`",
            f"- skipped trajectories: `{summary['num_skipped_trajectories']}`",
            f"- generated windows: `{summary['num_windows']}`",
            f"- window spec: `{summary['window_shape_spec']['context_len']}/{summary['window_shape_spec']['current_len']}/{summary['window_shape_spec']['future_len']}`",
            "",
            "## Boundaries",
            "",
            "- no cloud required now",
            "- no full BridgeData V2 download",
            "- no DROID download",
            "- no training",
            "- no VideoMAE token extraction",
            "- no importance generation",
            "- no VLM/RL/action-conditioned model",
            "",
            "## Metadata Policy",
            "",
            "- action saved as metadata but not used as input",
            "- language saved as metadata but not used as input",
            "- goal image saved as metadata but not used as input",
            "",
            "## Recommended Step21",
            "",
            f"- name: `{summary['recommended_step21']['name']}`",
            f"- condition: `{summary['recommended_step21']['condition']}`",
            f"- no_training: `{str(summary['recommended_step21']['no_training']).lower()}`",
            "",
        ]
    )


def render_step_doc(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# STEP20 BridgeData V2 Tiny-Subset Context Window Builder",
            "",
            "## 1. Goal",
            "",
            "Implement a BridgeData V2 tiny-subset window builder. This step does not train, download full data, extract tokens, or generate importance.",
            "",
            "## 2. Why BridgeData V2",
            "",
            "Step19 recommended BridgeData V2 as the first migration target and DROID as a future large-scale validation target.",
            "",
            "## 3. What Was Not Done",
            "",
            "- no full dataset download",
            "- no model download",
            "- no training",
            "- no token extraction",
            "- no importance generation",
            "- no action-conditioned model",
            "- no VLM/RL",
            "",
            "## 4. Input Policy",
            "",
            "- user-provided local tiny subset",
            "- manifest.jsonl",
            "- directory-per-trajectory inspection",
            "- fake manifest dry-run when the local subset is missing",
            "",
            "## 5. Manifest Format",
            "",
            "See `docs/BRIDGEDATA_V2_USER_SUBSET_FORMAT.md`.",
            "",
            "## 6. Window Spec",
            "",
            f"`context_len={summary['window_shape_spec']['context_len']}, current_len={summary['window_shape_spec']['current_len']}, future_len={summary['window_shape_spec']['future_len']}`",
            "",
            "## 7. LongContextSample Output",
            "",
            "Each output record is compatible with the Step19 `LongContextSample` schema and contains frame indices, optional frame path references, language/goal/action metadata references, and no video tensor.",
            "",
            "## 8. Metadata Policy",
            "",
            "Actions, language instructions, and goal images are saved as metadata only and are not model inputs.",
            "",
            "## 9. Outputs",
            "",
            "- window manifest JSONL",
            "- subset inspection summary",
            "- builder summary",
            "- eval report",
            "",
            "## 10. Tests",
            "",
            "Step20 adds targeted config, manifest, builder, missing-subset, no-download, artifact-blacklist, and report tests. Full pytest should remain green.",
            "",
            "## 11. Next Step",
            "",
            "Step21 should do BridgeData V2 tiny-subset token extraction dry-run only if the user provides a real tiny subset or explicitly approves a tiny download.",
            "",
        ]
    )


def render_user_subset_format_doc() -> str:
    return "\n".join(
        [
            "# BridgeData V2 User Subset Format",
            "",
            "Put user-provided tiny subsets under an ignored local-only directory:",
            "",
            "```text",
            "data/bridgedata_v2_tiny_user_subset/",
            "  manifest.jsonl",
            "  traj_000001/",
            "    images/",
            "      frame_000000.jpg",
            "      frame_000001.jpg",
            "    metadata.json",
            "    actions.npy or actions.json",
            "    goal.jpg optional",
            "```",
            "",
            "Each `manifest.jsonl` line should include:",
            "",
            "- trajectory_id",
            "- split",
            "- num_frames",
            "- frame_paths or image_dir",
            "- camera_names",
            "- actions_path",
            "- language_instruction",
            "- goal_image_path",
            "- task_id",
            "- environment_id",
            "- metadata",
            "",
            "The subset directory is ignored by git. Step20 only reads manifest metadata and file listings; it does not read image bytes or action arrays.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    args = parser.parse_args()
    summary = evaluate_window_builder(args.run_dir, args.docs_dir)
    print(json.dumps({"pass": summary["pass"], "num_windows": summary["num_windows"]}, indent=2))


if __name__ == "__main__":
    main()
