"""Inspect Step19 long-context dataset candidates without downloading data."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset_feasibility_types import DatasetFeasibilityEntry, LocalResourceSnapshot

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "long_context_dataset_feasibility_step19.yaml"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "long_context_dataset_feasibility_step19_v1"


def _gb(value: int) -> float:
    return round(value / (1024**3), 2)


def collect_local_resources(project_root: Path = PROJECT_ROOT) -> LocalResourceSnapshot:
    disk = shutil.disk_usage(project_root)
    ram_total = None
    ram_available = None
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        parsed: dict[str, int] = {}
        for line in meminfo.splitlines():
            key, rest = line.split(":", 1)
            parsed[key] = int(rest.strip().split()[0]) * 1024
        ram_total = _gb(parsed["MemTotal"])
        ram_available = _gb(parsed.get("MemAvailable", parsed["MemFree"]))
    except Exception:
        pass

    gpu_summary = "not checked"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        gpu_summary = result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "unavailable"
    except Exception:
        gpu_summary = "unavailable"

    return LocalResourceSnapshot(
        disk_free_gb=_gb(disk.free),
        disk_total_gb=_gb(disk.total),
        ram_total_gb=ram_total,
        ram_available_gb=ram_available,
        gpu_summary=gpu_summary,
    )


def _cache_exists(candidate: dict[str, Any]) -> bool:
    for path in candidate.get("local_cache_candidates", []):
        if Path(path).exists():
            return True
    return False


def _entry_for_candidate(candidate: dict[str, Any]) -> DatasetFeasibilityEntry:
    name = candidate["name"]
    local_cache_exists = _cache_exists(candidate)
    if name == "BridgeData V2":
        return DatasetFeasibilityEntry(
            dataset=name,
            priority=candidate.get("priority", "unknown"),
            full_download_allowed=False,
            local_cache_exists=local_cache_exists,
            supports_long_context_likely=True,
            supports_goal_image_likely=True,
            supports_language_likely=True,
            supports_action_likely=True,
            supports_multicam_likely=None,
            estimated_local_difficulty="medium",
            recommended_role="first local long-context migration target",
            recommended_next_action="metadata-only dry-run, then user-provided 100-500 trajectory tiny subset",
            evidence=[
                "official page describes goal-image and natural-language conditioning",
                "paper reports about 60k trajectories across 24 environments",
            ],
            unknowns=["exact local file layout and per-trajectory frame counts require a user-provided tiny subset"],
        )
    if name == "DROID":
        return DatasetFeasibilityEntry(
            dataset=name,
            priority=candidate.get("priority", "unknown"),
            full_download_allowed=False,
            local_cache_exists=local_cache_exists,
            supports_long_context_likely=True,
            supports_goal_image_likely=None,
            supports_language_likely=True,
            supports_action_likely=True,
            supports_multicam_likely=True,
            estimated_local_difficulty="high",
            recommended_role="future large-scale generalization validation target",
            recommended_next_action="defer full use until storage plan; keep Step19 metadata-only",
            evidence=[
                "official site reports about 76k trajectories and 350h interaction data",
                "DROID documentation describes RLDS and raw formats",
                "paper describes three camera views, robot state/action data, and language annotations",
            ],
            unknowns=["full local use needs storage planning and exact subset access policy"],
        )
    return DatasetFeasibilityEntry(
        dataset=name,
        priority=candidate.get("priority", "unknown"),
        full_download_allowed=False,
        local_cache_exists=local_cache_exists,
        supports_long_context_likely=None,
        supports_goal_image_likely=None,
        supports_language_likely=None,
        supports_action_likely=None,
        supports_multicam_likely=None,
        estimated_local_difficulty="unknown",
        recommended_role="optional candidate only",
        recommended_next_action="record as an open question; do not switch mainline without approval",
        unknowns=["candidate was not part of the Step19 mainline"],
    )


def inspect_candidates(config_path: Path = DEFAULT_CONFIG, output_dir: Path = DEFAULT_RUN_DIR) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    resources = collect_local_resources(PROJECT_ROOT)
    entries = [_entry_for_candidate(candidate).to_dict() for candidate in config["candidate_datasets"]]
    matrix = {
        "stage": config["stage"],
        "no_download": True,
        "full_download_performed": False,
        "large_download_performed": False,
        "training_performed": False,
        "resource_snapshot": resources.to_dict(),
        "datasets": entries,
        "optional_candidates": [
            {
                "dataset": "Open X-Embodiment / RT-X collection",
                "status": "listed only",
                "recommended_role": "future survey source, not Step20 mainline",
                "reason": "very broad and heterogeneous; useful later but too large for first local migration",
            }
        ],
    }
    (output_dir / "dataset_candidate_matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    (output_dir / "dataset_candidate_matrix.md").write_text(render_matrix_markdown(matrix), encoding="utf-8")
    return matrix


def render_matrix_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# Step19 Dataset Candidate Matrix",
        "",
        f"- no_download: `{str(matrix['no_download']).lower()}`",
        f"- training_performed: `{str(matrix['training_performed']).lower()}`",
        f"- disk_free_gb: `{matrix['resource_snapshot']['disk_free_gb']}`",
        f"- gpu_summary: `{matrix['resource_snapshot']['gpu_summary']}`",
        "",
        "| Dataset | Priority | Cache | Long context | Goal | Language | Action | Multicam | Difficulty | Role |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in matrix["datasets"]:
        lines.append(
            "| {dataset} | {priority} | {cache} | {long} | {goal} | {language} | {action} | {multicam} | {difficulty} | {role} |".format(
                dataset=item["dataset"],
                priority=item["priority"],
                cache=str(item["local_cache_exists"]).lower(),
                long=item["supports_long_context_likely"],
                goal=item["supports_goal_image_likely"],
                language=item["supports_language_likely"],
                action=item["supports_action_likely"],
                multicam=item["supports_multicam_likely"],
                difficulty=item["estimated_local_difficulty"],
                role=item["recommended_role"],
            )
        )
    lines.extend(["", "## Optional Candidates", ""])
    for item in matrix["optional_candidates"]:
        lines.append(f"- {item['dataset']}: {item['recommended_role']} ({item['reason']})")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    matrix = inspect_candidates(args.config, args.output_dir)
    print(json.dumps({"dataset_count": len(matrix["datasets"]), "no_download": matrix["no_download"]}, indent=2))


if __name__ == "__main__":
    main()
