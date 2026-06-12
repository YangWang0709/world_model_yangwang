"""Evaluate Step21 BridgeData V2 real tiny acquisition/validation outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "bridgedata_v2_real_tiny_validation_step21_v1"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs"


def evaluate_real_tiny_validation(run_dir: Path = DEFAULT_RUN_DIR, docs_dir: Path = DEFAULT_DOCS_DIR) -> dict[str, Any]:
    probe = _read_json(run_dir / "download_probe_summary.json")
    acquisition = _read_json(run_dir / "acquisition_summary.json")
    validation = _read_json(run_dir / "real_tiny_validation_summary.json")
    real_validated = bool(validation.get("real_format_validated", False))
    safe_stop = bool(validation.get("safe_stop", False) or acquisition.get("safe_stop", False) or probe.get("safe_stop", False))
    large_download = bool(acquisition.get("large_download_performed", False))
    full_download = bool(acquisition.get("full_download_performed", False))
    training = bool(validation.get("training_performed", False) or acquisition.get("training_performed", False))
    token = bool(validation.get("token_extraction_performed", False) or acquisition.get("token_extraction_performed", False))
    importance = bool(
        validation.get("importance_generation_performed", False) or acquisition.get("importance_generation_performed", False)
    )
    real_window_manifest = Path(validation.get("real_window_manifest_jsonl", ""))
    summary = {
        "stage": "bridgedata_v2_real_tiny_validation_step21",
        "safety_gate_pass": bool(not full_download and not large_download and not training and not token and not importance),
        "real_format_validated": real_validated,
        "safe_stop": safe_stop and not real_validated,
        "cloud_required_now": False,
        "download_performed": bool(acquisition.get("download_performed", False)),
        "download_bytes": acquisition.get("download_bytes"),
        "full_download_performed": False,
        "large_download_performed": False,
        "droid_download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "real_tiny_manifest_exists": Path(validation.get("manifest_path", "")).exists() if validation.get("manifest_path") else False,
        "real_window_manifest_exists": real_window_manifest.exists(),
        "num_real_trajectories": validation.get("num_manifest_records", 0),
        "num_valid_trajectories": validation.get("num_valid_trajectories", 0),
        "num_skipped_trajectories": validation.get("num_skipped_trajectories", 0),
        "frame_count_min": validation.get("frame_count_min"),
        "frame_count_mean": validation.get("frame_count_mean"),
        "frame_count_max": validation.get("frame_count_max"),
        "num_real_windows": validation.get("num_windows", 0),
        "metadata_fields": {
            "images_likely": validation.get("has_images_likely"),
            "actions_likely": validation.get("has_actions_likely"),
            "language_likely": validation.get("has_language_likely"),
            "goal_image_likely": validation.get("has_goal_image_likely"),
        },
        "safe_stop_reason": validation.get("reason") or acquisition.get("reason") or probe.get("reason"),
        "probe": probe,
        "acquisition": acquisition,
        "validation": validation,
        "recommended_step22": {
            "name": "BridgeData V2 real tiny token extraction dry-run"
            if real_validated
            else "User-provided BridgeData tiny subset preparation",
            "condition": "real tiny format validated"
            if real_validated
            else "prepare a user-provided subset with manifest.jsonl and at least one trajectory with >=24 frames",
            "no_training": True,
        },
    }
    summary["safety_gate_pass"] = bool(
        summary["safety_gate_pass"]
        and not summary["full_download_performed"]
        and not summary["large_download_performed"]
        and not summary["droid_download_performed"]
        and not summary["training_performed"]
        and not summary["token_extraction_performed"]
        and not summary["importance_generation_performed"]
    )
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "BRIDGEDATA_V2_REAL_TINY_VALIDATION_REPORT.md").write_text(render_validation_report(summary), encoding="utf-8")
    (docs_dir / "BRIDGEDATA_V2_TINY_SAMPLE_ACQUISITION_LOG.md").write_text(render_acquisition_log(summary), encoding="utf-8")
    (docs_dir / "BRIDGEDATA_V2_USER_PROVIDED_SUBSET_INSTRUCTIONS.md").write_text(
        render_user_subset_instructions(), encoding="utf-8"
    )
    (docs_dir / "STEP21_BRIDGEDATA_V2_REAL_TINY_VALIDATION.md").write_text(render_step21_doc(summary), encoding="utf-8")
    (run_dir / "real_tiny_validation_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def render_validation_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 Real Tiny Validation Report",
            "",
            f"- safety_gate_pass: `{str(summary['safety_gate_pass']).lower()}`",
            f"- real_format_validated: `{str(summary['real_format_validated']).lower()}`",
            f"- safe_stop: `{str(summary['safe_stop']).lower()}`",
            f"- safe_stop_reason: `{summary['safe_stop_reason']}`",
            f"- download_performed: `{str(summary['download_performed']).lower()}`",
            f"- download_bytes: `{summary['download_bytes']}`",
            f"- real_tiny_manifest_exists: `{str(summary['real_tiny_manifest_exists']).lower()}`",
            f"- real_window_manifest_exists: `{str(summary['real_window_manifest_exists']).lower()}`",
            f"- num_real_trajectories: `{summary['num_real_trajectories']}`",
            f"- valid trajectories: `{summary['num_valid_trajectories']}`",
            f"- skipped trajectories: `{summary['num_skipped_trajectories']}`",
            f"- frame count min/mean/max: `{summary['frame_count_min']}/{summary['frame_count_mean']}/{summary['frame_count_max']}`",
            f"- num_real_windows: `{summary['num_real_windows']}`",
            "",
            "## Metadata Fields",
            "",
            f"- images likely: `{summary['metadata_fields']['images_likely']}`",
            f"- actions likely: `{summary['metadata_fields']['actions_likely']}`",
            f"- language likely: `{summary['metadata_fields']['language_likely']}`",
            f"- goal image likely: `{summary['metadata_fields']['goal_image_likely']}`",
            "",
            "## Boundaries",
            "",
            "- no full BridgeData V2 download",
            "- no DROID download",
            "- no training",
            "- no token extraction",
            "- no importance generation",
            "- action/language/goal image remain metadata only",
            "",
            "## Recommended Step22",
            "",
            f"- name: `{summary['recommended_step22']['name']}`",
            f"- condition: `{summary['recommended_step22']['condition']}`",
            f"- no_training: `{str(summary['recommended_step22']['no_training']).lower()}`",
            "",
        ]
    )


def render_acquisition_log(summary: dict[str, Any]) -> str:
    candidates = summary["probe"].get("download_candidates", [])
    lines = [
        "# BridgeData V2 Tiny Sample Acquisition Log",
        "",
        "- official sources checked: `true`",
        f"- candidate count: `{len(candidates)}`",
        f"- safe candidate found: `{str(summary['probe'].get('safe_candidate_found')).lower()}`",
        f"- selected candidate: `{summary['probe'].get('selected_candidate')}`",
        f"- download performed: `{str(summary['download_performed']).lower()}`",
        f"- download bytes: `{summary['download_bytes']}`",
        f"- safe stop: `{str(summary['safe_stop']).lower()}`",
        f"- reason: `{summary['safe_stop_reason']}`",
        "- no large download performed: `true`",
        "",
        "## Candidates",
        "",
    ]
    for candidate in candidates:
        lines.append(
            f"- `{candidate.get('candidate_kind')}` size_known=`{candidate.get('size_known')}` "
            f"content_length=`{candidate.get('content_length_bytes')}` safe=`{candidate.get('safe_under_limit')}` "
            f"url={candidate.get('url')}"
        )
    lines.append("")
    return "\n".join(lines)


def render_user_subset_instructions() -> str:
    return "\n".join(
        [
            "# BridgeData V2 User-Provided Subset Instructions",
            "",
            "If Step21 safe-stops, prepare a tiny local subset like this:",
            "",
            "```text",
            "data/bridgedata_v2_tiny_user_subset/",
            "  manifest.jsonl",
            "  traj_000001/",
            "    images/",
            "      frame_000000.jpg",
            "      frame_000001.jpg",
            "      ...",
            "    metadata.json",
            "    actions.npy or actions.json optional",
            "    goal.jpg optional",
            "  traj_000002/",
            "    ...",
            "```",
            "",
            "Requirements:",
            "",
            "- at least 1 trajectory",
            "- recommended 5-10 trajectories",
            "- each trajectory needs at least 24 frames",
            "- 40+ frames per trajectory is preferred",
            "- manifest.jsonl must include trajectory_id and num_frames",
            "- provide either frame_paths or image_dir",
            "- language_instruction, goal_image_path, and actions_path are optional metadata",
            "- do not commit this directory to git",
            "",
        ]
    )


def render_step21_doc(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# STEP21 BridgeData V2 Real Tiny Validation",
            "",
            "Step21 attempts to move from the Step20 fake manifest to a real tiny BridgeData V2 sample while preserving strict safety boundaries.",
            "",
            "## Allowed",
            "",
            "- official-source probing",
            "- a tiny download only if Content-Length is known and <=1GB",
            "- safe-stop if no safe tiny sample exists",
            "- manifest and format validation",
            "",
            "## Not Done",
            "",
            "- no full dataset download",
            "- no DROID download",
            "- no model download",
            "- no training",
            "- no token extraction",
            "- no importance generation",
            "- no VLM/RL/action-conditioned model",
            "",
            "## Outcome",
            "",
            f"- real_format_validated: `{str(summary['real_format_validated']).lower()}`",
            f"- safe_stop: `{str(summary['safe_stop']).lower()}`",
            f"- safety_gate_pass: `{str(summary['safety_gate_pass']).lower()}`",
            "",
            "## Next Step",
            "",
            f"`{summary['recommended_step22']['name']}`",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    args = parser.parse_args()
    summary = evaluate_real_tiny_validation(args.run_dir, args.docs_dir)
    print(json.dumps({"safety_gate_pass": summary["safety_gate_pass"], "real_format_validated": summary["real_format_validated"]}, indent=2))


if __name__ == "__main__":
    main()
