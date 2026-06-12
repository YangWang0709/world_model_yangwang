"""Evaluate Step19 long-context dataset feasibility dry-run outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "long_context_dataset_feasibility_step19_v1"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_feasibility_summary(
    run_dir: Path = DEFAULT_RUN_DIR,
    docs_dir: Path = DEFAULT_DOCS_DIR,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    matrix = _read_json(run_dir / "dataset_candidate_matrix.json")
    bridge = _read_json(run_dir / "bridgedata_v2_dryrun_summary.json")
    droid = _read_json(run_dir / "droid_dryrun_summary.json")
    summary = {
        "stage": "long_context_dataset_feasibility_step19",
        "cloud_required_now": False,
        "full_download_performed": False,
        "large_download_performed": False,
        "training_performed": False,
        "model_download_performed": False,
        "current_project_conclusion": {
            "step17_context_pipeline_valid": True,
            "step18_loss_ablation_negative": True,
            "step17_5_audit_pass": True,
            "bair_limitations": [
                "BAIR clips are short, so current 4 frames are already competitive.",
                "Step18 context importance labels were diffuse, so sparse token selection is weak.",
                "Selector-level gains did not convert into downstream future-MSE gains.",
            ],
        },
        "dataset_recommendation": {
            "first_migration_target": "BridgeData V2",
            "future_large_scale_target": "DROID",
            "optional_future_survey": "Open X-Embodiment / RT-X collection",
            "reason": [
                "BridgeData V2 is closer to a local tiny-subset migration and supports goal/language-conditioned future work.",
                "DROID is larger and richer, but storage and format planning should come before local full use.",
                "Open X-Embodiment is valuable as a future source of datasets but too broad for the first Step20 mainline.",
            ],
        },
        "recommended_step20": {
            "name": "BridgeData V2 tiny-subset context window builder",
            "download_policy": "metadata-or-user-provided-small-subset-only",
            "train": False,
            "target_subset_size": "100-500 trajectories",
            "context_len": 16,
            "current_len": 4,
            "future_len": 4,
            "importance_level": "temporal/block-level first",
            "encoder": "existing local VideoMAE first",
            "use_action_as_input": False,
            "save_action_language_goal_as_metadata": True,
        },
        "candidate_matrix": matrix["datasets"],
        "resource_snapshot": matrix["resource_snapshot"],
        "bridge_dryrun": {
            "no_download": bridge["no_download"],
            "recommended_initial_subset": bridge["recommended_initial_subset"],
            "recommended_next_step": bridge["recommended_next_step"],
        },
        "droid_dryrun": {
            "no_download": droid["no_download"],
            "recommended_role": droid["recommended_role"],
            "recommended_next_step": droid["recommended_next_step"],
        },
        "temporal_block_importance_design": [
            "keep all current tokens intact",
            "group context by frame, temporal segment, and optional spatial blocks",
            "score temporal blocks before individual tokens to reduce diffuse-label noise",
            "aggregate selected context with block-aware pooling instead of plain mean_pool_selected_context only",
            "continue to save actions/language/goal as metadata, not model inputs",
        ],
        "risks": [
            "official dataset format details still need tiny-subset verification",
            "BridgeData V2 full download is not appropriate for this local Step19 scope",
            "DROID full use likely requires external storage or cloud planning",
            "language/goal metadata could tempt a VLM path, but Step20 should keep it metadata-only",
        ],
        "open_questions": [
            "exact BridgeData V2 tiny-subset access path and file layout",
            "per-trajectory frame-count distribution after frame-rate subsampling",
            "whether goal-image fields are consistently present in the chosen BridgeData V2 subset",
            "which DROID format, RLDS or raw, should be used if future storage is approved",
        ],
    }
    summary["sanity_gate_pass"] = bool(
        not summary["cloud_required_now"]
        and not summary["full_download_performed"]
        and not summary["large_download_performed"]
        and not summary["training_performed"]
        and bridge["no_download"]
        and droid["no_download"]
    )
    (run_dir / "long_context_dataset_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (run_dir / "long_context_dataset_feasibility_summary.md").write_text(
        render_summary_markdown(summary), encoding="utf-8"
    )
    write_docs(summary, docs_dir)
    return summary


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Step19 Long-Context Dataset Feasibility Summary",
        "",
        f"- cloud_required_now: `{str(summary['cloud_required_now']).lower()}`",
        f"- full_download_performed: `{str(summary['full_download_performed']).lower()}`",
        f"- large_download_performed: `{str(summary['large_download_performed']).lower()}`",
        f"- training_performed: `{str(summary['training_performed']).lower()}`",
        f"- sanity_gate_pass: `{str(summary['sanity_gate_pass']).lower()}`",
        "",
        "## Recommendation",
        "",
        f"- first migration target: `{summary['dataset_recommendation']['first_migration_target']}`",
        f"- future large-scale target: `{summary['dataset_recommendation']['future_large_scale_target']}`",
        f"- recommended Step20: `{summary['recommended_step20']['name']}`",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["dataset_recommendation"]["reason"])
    lines.extend(["", "## Risks", ""])
    lines.extend(f"- {item}" for item in summary["risks"])
    lines.extend(["", "## Open Questions", ""])
    lines.extend(f"- {item}" for item in summary["open_questions"])
    lines.append("")
    return "\n".join(lines)


def write_docs(summary: dict[str, Any], docs_dir: Path) -> None:
    (docs_dir / "LONG_CONTEXT_DATASET_FEASIBILITY_REPORT.md").write_text(
        render_feasibility_report(summary), encoding="utf-8"
    )
    (docs_dir / "STEP19_LONG_CONTEXT_DATASET_FEASIBILITY.md").write_text(
        render_step_doc(summary), encoding="utf-8"
    )
    (docs_dir / "BRIDGEDATA_V2_MIGRATION_PLAN.md").write_text(
        render_bridge_plan(summary), encoding="utf-8"
    )
    (docs_dir / "DROID_MIGRATION_PLAN.md").write_text(render_droid_plan(summary), encoding="utf-8")


def render_feasibility_report(summary: dict[str, Any]) -> str:
    matrix_lines = [
        "| Dataset | Role | Difficulty | Next action |",
        "|---|---|---|---|",
    ]
    for item in summary["candidate_matrix"]:
        matrix_lines.append(
            f"| {item['dataset']} | {item['recommended_role']} | {item['estimated_local_difficulty']} | {item['recommended_next_action']} |"
        )
    return "\n".join(
        [
            "# Long-Context Dataset Feasibility Report",
            "",
            "Step19 is a feasibility and migration-planning step only. It performed no training, no model download, and no full dataset download.",
            "",
            "## Current Project Conclusion",
            "",
            "- Step17 validated the context bottleneck pipeline.",
            "- Step18 showed that BAIR selector-loss changes did not improve downstream future MSE.",
            "- Step17.5 audit passed, reducing the chance that the negative result is a simple implementation bug.",
            "- Therefore, continuing BAIR loss ablations is lower value than moving to a longer-context dataset.",
            "",
            "## Candidate Matrix",
            "",
            *matrix_lines,
            "",
            "## Recommendation",
            "",
            f"- first migration target: `{summary['dataset_recommendation']['first_migration_target']}`",
            f"- future large-scale target: `{summary['dataset_recommendation']['future_large_scale_target']}`",
            f"- Step20: `{summary['recommended_step20']['name']}`",
            "",
            "## Source Notes",
            "",
            "- BridgeData V2 official page: https://rail-berkeley.github.io/bridgedata/",
            "- BridgeData V2 paper page: https://proceedings.mlr.press/v229/walke23a.html",
            "- BridgeData V2 official page/paper reports a large robot manipulation dataset with about 60k trajectories across 24 environments and goal/language-conditioned use cases.",
            "- DROID official page: https://droid-dataset.github.io/",
            "- DROID dataset docs: https://droid-dataset.github.io/droid/the-droid-dataset",
            "- DROID paper page: https://arxiv.org/abs/2403.12945",
            "- DROID official page/paper reports about 76k demonstration trajectories, 350h interaction data, 564 scenes, 86 tasks, and multi-camera data.",
            "- Open X-Embodiment official page: https://robotics-transformer-x.github.io/",
            "- Open X-Embodiment is listed only as a future survey source because it is too broad and large for the first local migration.",
            "",
            "## Sanity Gates",
            "",
            f"- cloud_required_now: `{str(summary['cloud_required_now']).lower()}`",
            f"- full_download_performed: `{str(summary['full_download_performed']).lower()}`",
            f"- large_download_performed: `{str(summary['large_download_performed']).lower()}`",
            f"- training_performed: `{str(summary['training_performed']).lower()}`",
            f"- sanity_gate_pass: `{str(summary['sanity_gate_pass']).lower()}`",
            "",
        ]
    )


def render_step_doc(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# STEP19 Long-Context Dataset Feasibility",
            "",
            "## 1. Why Step19",
            "",
            "Step17 showed that the context bottleneck pipeline works. Step18 showed a negative BAIR loss-ablation result. Step17.5 audited the code path and passed. The likely issue is now dataset/task suitability: BAIR is too short and current-only context is already competitive.",
            "",
            "## 2. What Was Not Done",
            "",
            "- no training",
            "- no full dataset download",
            "- no model download",
            "- no BAIR rerun",
            "- no VLM/RL/action-conditioned model",
            "",
            "## 3. Dataset Candidates",
            "",
            "- BridgeData V2",
            "- DROID",
            "- Open X-Embodiment / RT-X collection as a listed-only future survey source",
            "",
            "## 4. Feasibility Criteria",
            "",
            "- trajectory length",
            "- image observations",
            "- action metadata",
            "- language/goal metadata",
            "- multi-camera metadata",
            "- local subset feasibility",
            "- storage requirement",
            "- suitability for temporal/block context importance",
            "",
            "## 5. Unified LongContextSample Schema",
            "",
            "The schema records dataset name, split, trajectory/sample ids, context/current/future frame indices, optional video tensors, optional actions, end-effector state, language instruction, goal image, camera names, and metadata. Missing dataset-specific fields are allowed to be None.",
            "",
            "## 6. Window Design",
            "",
            "- short control: context_len=8, current_len=4, future_len=4",
            "- medium: context_len=16, current_len=4, future_len=4",
            "- long: context_len=32, current_len=4, future_len=8",
            "",
            "## 7. BridgeData V2 Migration Plan",
            "",
            "Use BridgeData V2 as the first migration target with a user-provided or metadata-only tiny subset of 100-500 trajectories. Keep actions, language, and goal images as metadata first.",
            "",
            "## 8. DROID Migration Plan",
            "",
            "Keep DROID as a future large-scale generalization target. It is richer and broader, but full use needs storage planning.",
            "",
            "## 9. Recommendation",
            "",
            f"First target: {summary['dataset_recommendation']['first_migration_target']}. Future large-scale target: {summary['dataset_recommendation']['future_large_scale_target']}.",
            "",
            "## 10. Risks",
            "",
            *[f"- {item}" for item in summary["risks"]],
            "",
            "## 11. Next Step",
            "",
            "Step20 should build a BridgeData V2 tiny-subset context window builder with no training, context_len=16, current_len=4, future_len=4, and temporal/block-level importance design.",
            "",
        ]
    )


def render_bridge_plan(summary: dict[str, Any]) -> str:
    subset = summary["recommended_step20"]
    return "\n".join(
        [
            "# BridgeData V2 Migration Plan",
            "",
            "## Role",
            "",
            "BridgeData V2 is recommended as the first long-context migration target.",
            "",
            "## Why",
            "",
            "- broader and longer than BAIR for manipulation behavior",
            "- closer to local tiny-subset feasibility than DROID",
            "- goal image and language metadata can support later task-grounded conditioning",
            "- Step20 still does not connect VLM or a language encoder",
            "",
            "## Initial Subset",
            "",
            f"- trajectories: `{subset['target_subset_size']}`",
            f"- context_len: `{subset['context_len']}`",
            f"- current_len: `{subset['current_len']}`",
            f"- future_len: `{subset['future_len']}`",
            f"- encoder: `{subset['encoder']}`",
            "- action/language/goal: metadata only",
            "- training: `false`",
            "",
            "## Required Answers Before Full Migration",
            "",
            "- exact tiny-subset source path",
            "- per-trajectory frame count distribution",
            "- consistency of goal-image and language fields",
            "- final local disk budget for preprocessed windows and tokens",
            "",
        ]
    )


def render_droid_plan(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# DROID Migration Plan",
            "",
            "## Role",
            "",
            "DROID is recommended as a future large-scale validation target, not the first local migration target.",
            "",
            "## Why",
            "",
            "- much larger and more diverse than BAIR or a BridgeData tiny subset",
            "- useful for later generalization claims",
            "- full local use likely needs storage planning, external disk, or cloud resources",
            "- format choice between RLDS and raw data should be decided later",
            "",
            "## Step19 Decision",
            "",
            "- no full download",
            "- no real sample download",
            "- no training",
            "- keep only schema mapping and feasibility notes",
            "",
            "## Future Preconditions",
            "",
            "- approved storage plan",
            "- approved subset policy",
            "- explicit decision on raw versus RLDS format",
            "- separate no-leakage gate for language/action metadata if they become model inputs later",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    args = parser.parse_args()
    summary = write_feasibility_summary(args.run_dir, args.docs_dir)
    print(json.dumps({"sanity_gate_pass": summary["sanity_gate_pass"]}, indent=2))


if __name__ == "__main__":
    main()
