"""End-to-end Step 13 smoke for BAIR selector loss ablation."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bair_selector_ablation import (
    load_selector_ablation_rows,
    write_selector_ablation_summaries,
)
from eval.eval_bair_selector_downstream_ablation import (
    load_downstream_ablation_rows,
    write_downstream_ablation_summaries,
)
from training.run_bair_selector_ablation import validate_selector_ablation_inputs
from training.run_baseline_comparison import load_yaml
from training.selector_ablation_trainer import run_selector_ablation


CONFIG_PATH = PROJECT_ROOT / "configs" / "selector_ablation_bair_videomae_smoke.yaml"
REPORT_PATH = PROJECT_ROOT / "docs" / "BAIR_SELECTOR_ABLATION_SMOKE_REPORT.md"
STEP13_DOC_PATH = PROJECT_ROOT / "docs" / "STEP13_BAIR_SELECTOR_ABLATION.md"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _ram_snapshot() -> dict[str, float]:
    values: dict[str, float] = {}
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {"ram_used_gib": 0.0, "ram_total_gib": 0.0}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, raw_value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = float(raw_value.strip().split()[0]) * 1024.0
    total = values.get("MemTotal", 0.0)
    available = values.get("MemAvailable", 0.0)
    gib = 1024.0**3
    return {
        "ram_used_gib": (total - available) / gib if total else 0.0,
        "ram_total_gib": total / gib if total else 0.0,
    }


def _gpu_snapshot() -> dict[str, float | str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return {"gpu_name": "unavailable", "gpu_mem_total_gib": 0.0, "gpu_mem_used_gib": 0.0}
    first = result.stdout.strip().splitlines()[0]
    name, total_mib, used_mib = [part.strip() for part in first.split(",")]
    return {
        "gpu_name": name,
        "gpu_mem_total_gib": float(total_mib) / 1024.0,
        "gpu_mem_used_gib": float(used_mib) / 1024.0,
    }


def _resource_snapshot() -> dict[str, Any]:
    return {**_ram_snapshot(), **_gpu_snapshot()}


def _clean_run_dir(config: dict[str, Any]) -> Path:
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    expected = (PROJECT_ROOT / "runs" / "selector_ablation_bair_videomae_smoke_v1").resolve()
    resolved = run_dir.resolve()
    if resolved != expected:
        raise ValueError(f"Refusing to clean unexpected run dir: {run_dir}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return run_dir


def _read(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""


def _best_name(aggregate: dict[str, Any], key: str) -> str:
    row = aggregate.get(key) or {}
    return str(row.get("variant_name", "n/a"))


def _write_smoke_report(
    config: dict[str, Any],
    result: dict[str, Any],
    resources: dict[str, Any],
    checks: dict[str, bool],
    pass_flag: bool,
) -> None:
    selector_md = _read(result["selector_summary_md"])
    downstream_md = _read(result["downstream_summary_md"])
    combined_md = _read(result["combined_report_md"])
    selector_aggregate = result["selector_aggregate"]
    downstream_aggregate = result["downstream_aggregate"]
    comparison = downstream_aggregate.get("comparison", {})
    command = (
        "/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab "
        "python scripts/smoke_test_bair_selector_ablation.py"
    )
    lines = [
        "# BAIR Selector Ablation Smoke Report",
        "",
        f"Command: `{command}`",
        "",
        f"- run dir: `{result['run_dir']}`",
        f"- selector variants: `{', '.join(result['loss_variants'])}`",
        f"- selector summary: `{result['selector_summary_json']}`",
        f"- downstream summary: `{result['downstream_summary_json']}`",
        f"- combined report: `{result['combined_report_json']}`",
        "",
        "## Selector-Level Summary",
        "",
        selector_md,
        "",
        "## Downstream-Level Summary",
        "",
        downstream_md,
        "",
        "## Comparison With Step 12",
        "",
        combined_md,
        "",
        "## Best Variants",
        "",
        f"- best selector by selected importance: `{_best_name(selector_aggregate, 'best_by_selected_teacher_importance')}`",
        f"- best selector by topK overlap: `{_best_name(selector_aggregate, 'best_by_target_topk_overlap')}`",
        f"- best downstream by MSE: `{_best_name(downstream_aggregate, 'best_by_student_future_mse')}`",
        "",
        "## Improvement Flags",
        "",
        "```json",
        json.dumps(comparison, indent=2),
        "```",
        "",
        "## Resource Usage",
        "",
        "```json",
        json.dumps(resources, indent=2),
        "```",
        "",
        "## Checks",
        "",
        "```json",
        json.dumps(checks, indent=2),
        "```",
        "",
        f"BAIR_SELECTOR_ABLATION_SMOKE_PASS = {str(pass_flag).lower()}",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _next_step_recommendation(comparison: dict[str, Any]) -> str:
    if (
        comparison.get("downstream_improvement_over_step12_learned")
        and comparison.get("any_variant_selected_importance_gt_step12_learned")
        and comparison.get("any_variant_topk_overlap_gt_step12_learned")
    ):
        return "Step 14 can scale BAIR to a larger split such as train 500 / test 64."
    if comparison.get("any_variant_selected_importance_gt_step12_learned") or comparison.get(
        "any_variant_topk_overlap_gt_step12_learned"
    ):
        return "Step 14 should optimize TokenCompressor / StudentWorldModel because selector-level gains did not fully translate downstream."
    return "Step 14 should run more seeds or revisit Teacher importance quality before scaling data."


def _write_step13_doc(
    config: dict[str, Any],
    result: dict[str, Any],
    resources: dict[str, Any],
    pass_flag: bool,
    pytest_result: str = "Run after Step 13 smoke; final result is recorded in STEP13_LOCAL_SUMMARY.md and the final response.",
) -> None:
    selector_md = _read(result["selector_summary_md"])
    downstream_md = _read(result["downstream_summary_md"])
    combined = result["combined_report"]
    comparison = result["downstream_aggregate"].get("comparison", {})
    step12_reference = result["downstream_aggregate"].get("step12_reference", {})
    doc = [
        "# STEP13 BAIR Selector Loss / TopK Supervision Ablation",
        "",
        "## 1. Goal",
        "",
        "Compare selector loss and topK-supervision variants on BAIR Robot Pushing small with existing VideoMAE token shards.",
        "",
        "## 2. Motivation",
        "",
        "Step 12 showed that the learned selector beats Random-K, while Teacher-Importance TopK remains a better upper bound and Uniform-K is slightly better than the Step 12 learned selector on downstream MSE. Step 13 strengthens selector supervision before scaling data.",
        "",
        "## 3. Cloud Server Decision",
        "",
        "No cloud server is required for this stage. The run reads existing BAIR token and importance shards and does not run VideoMAE, Teacher training, or importance generation.",
        "",
        f"- OOM: `{resources.get('oom', False)}`",
        f"- max RAM used GiB: `{resources.get('max_ram_used_gib')}`",
        f"- max GPU memory used GiB: `{resources.get('max_gpu_mem_used_gib')}`",
        f"- cloud recommendation: `{resources.get('cloud_recommendation')}`",
        "",
        "## 4. Input",
        "",
        f"- train token dir: `{config['data']['train_token_shard_dir']}`",
        f"- test token dir: `{config['data']['test_token_shard_dir']}`",
        f"- train importance dir: `{config['data']['train_importance_shard_dir']}`",
        f"- test importance dir: `{config['data']['test_importance_shard_dir']}`",
        f"- Step 12 baseline summary: `{config['step12_reference']['baseline_summary_json']}`",
        f"- train/test samples: `{config['data']['max_train_samples']} / {config['data']['max_test_samples']}`",
        "- token shape: `[B, 392, 768]`",
        "- source encoder: `VideoMAE`",
        "",
        "## 5. Loss Variants",
        "",
        "- mse_only",
        "- weighted_mse",
        "- mse_plus_pairwise_rank",
        "- topk_bce",
        "- hybrid_weighted_mse_rank_bce",
        "",
        "## 6. Selector-Level Results",
        "",
        selector_md,
        "",
        "## 7. Downstream StudentWorldModel Results",
        "",
        downstream_md,
        "",
        "## 8. Comparison With Step 12",
        "",
        "```json",
        json.dumps(step12_reference, indent=2),
        "```",
        "",
        "```json",
        json.dumps(comparison, indent=2),
        "```",
        "",
        "## 9. Sanity Gate",
        "",
        "```json",
        json.dumps(combined.get("sanity_gate", {}), indent=2),
        "```",
        "",
        "## 10. Pytest Result",
        "",
        pytest_result,
        "",
        "## 11. Git Commit",
        "",
        "Branch: `feature/tgpawb-step3-token-extraction`. The final commit hash and push status are recorded after commit/push in `STEP13_LOCAL_SUMMARY.md` and the final response. No PR is created.",
        "",
        "## 12. What Was Not Done",
        "",
        "- no new model download",
        "- no new dataset download",
        "- no BAIR re-download",
        "- no VideoMAE training",
        "- no Teacher retraining",
        "- no Step11E selector overwrite",
        "- no Step11F StudentWorldModel overwrite",
        "- no Step12 baseline overwrite",
        "- no VLM grounding",
        "- no RL / policy optimization",
        "- no model weight committed",
        "- no BAIR data committed",
        "- no token shard committed",
        "- no importance shard committed",
        "- no checkpoint committed",
        "- no password or token saved",
        "- no PR created",
        "",
        "## 13. Next Step Recommendation",
        "",
        _next_step_recommendation(comparison),
        "",
        f"BAIR_SELECTOR_ABLATION_SMOKE_PASS = {str(pass_flag).lower()}",
    ]
    STEP13_DOC_PATH.write_text("\n".join(doc) + "\n", encoding="utf-8")


def _failure_payload(error: BaseException, resources: dict[str, Any], elapsed_time_sec: float) -> None:
    payload = {
        "error": repr(error),
        "traceback": traceback.format_exc(),
        "resources": resources,
        "elapsed_time_sec": elapsed_time_sec,
        "oom": "out of memory" in repr(error).lower(),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# BAIR Selector Ablation Smoke Report\n\n"
        "Step 13 failed before producing a complete smoke report.\n\n"
        "```json\n"
        + json.dumps(payload, indent=2)
        + "\n```\n\n"
        "BAIR_SELECTOR_ABLATION_SMOKE_PASS = false\n",
        encoding="utf-8",
    )


def main() -> None:
    start = time.time()
    before = _resource_snapshot()
    config = load_yaml(CONFIG_PATH)
    resources: dict[str, Any] = {"before": before}
    try:
        validate_selector_ablation_inputs(config)
        run_dir = _clean_run_dir(config)
        result = run_selector_ablation(config)

        selector_rows = load_selector_ablation_rows(run_dir)
        selector_paths = write_selector_ablation_summaries(selector_rows, run_dir=run_dir)
        downstream_rows = load_downstream_ablation_rows(run_dir)
        downstream_paths = write_downstream_ablation_summaries(
            downstream_rows,
            run_dir=run_dir,
            step12_summary_json=config["step12_reference"]["baseline_summary_json"],
            selector_summary={
                "rows": selector_rows,
                "aggregate": selector_paths["selector_aggregate"],
            },
        )
        result.update(selector_paths)
        result.update(downstream_paths)

        after = _resource_snapshot()
        elapsed = time.time() - start
        resources["after"] = after
        resources["elapsed_time_sec"] = elapsed
        resources["max_ram_used_gib"] = max(before.get("ram_used_gib", 0.0), after.get("ram_used_gib", 0.0))
        resources["max_gpu_mem_used_gib"] = max(
            before.get("gpu_mem_used_gib", 0.0),
            after.get("gpu_mem_used_gib", 0.0),
        )
        resources["oom"] = False
        max_ram_gb_warning = float(config.get("resource_limits", {}).get("max_ram_gb_warning", 28.0))
        max_gpu_mem_fraction_warning = float(
            config.get("resource_limits", {}).get("max_gpu_mem_fraction_warning", 0.9)
        )
        gpu_fraction = (
            resources["max_gpu_mem_used_gib"] / after["gpu_mem_total_gib"]
            if after.get("gpu_mem_total_gib", 0.0)
            else 0.0
        )
        resources["gpu_mem_fraction"] = gpu_fraction
        resources["cloud_recommendation"] = (
            "consider 4090 / 48GB when scaling BAIR beyond this smoke"
            if resources["max_ram_used_gib"] >= max_ram_gb_warning or gpu_fraction >= max_gpu_mem_fraction_warning
            else "not required for Step 13 smoke"
        )

        selector_payload = json.loads(Path(result["selector_summary_json"]).read_text(encoding="utf-8"))
        downstream_payload = json.loads(Path(result["downstream_summary_json"]).read_text(encoding="utf-8"))
        combined_payload = json.loads(Path(result["combined_report_json"]).read_text(encoding="utf-8"))
        successful_selectors = [row for row in selector_payload["rows"] if row.get("success", True)]
        successful_downstream = [row for row in downstream_payload["rows"] if row.get("success", True)]
        checks = {
            "selector_summary_json_exists": Path(result["selector_summary_json"]).exists(),
            "selector_summary_csv_exists": Path(result["selector_summary_csv"]).exists(),
            "selector_summary_md_exists": Path(result["selector_summary_md"]).exists(),
            "downstream_summary_json_exists": Path(result["downstream_summary_json"]).exists(),
            "downstream_summary_csv_exists": Path(result["downstream_summary_csv"]).exists(),
            "downstream_summary_md_exists": Path(result["downstream_summary_md"]).exists(),
            "combined_report_json_exists": Path(result["combined_report_json"]).exists(),
            "combined_report_md_exists": Path(result["combined_report_md"]).exists(),
            "mse_only_variant_success": any(row.get("loss_type") == "mse_only" for row in successful_selectors),
            "non_mse_only_variant_success": any(
                row.get("loss_type") != "mse_only" for row in successful_selectors
            ),
            "downstream_result_finite": any(_finite(row.get("student_future_mse")) for row in successful_downstream),
            "selector_sanity_gate_pass": bool(selector_payload["aggregate"]["sanity_gate"]["pass"]),
            "downstream_sanity_gate_pass": bool(downstream_payload["aggregate"]["sanity_gate"]["pass"]),
            "combined_sanity_gate_pass": bool(combined_payload["sanity_gate"]["pass"]),
            "ram_below_warning": resources["max_ram_used_gib"] < max_ram_gb_warning,
            "oom_false": resources["oom"] is False,
        }
        pass_flag = all(checks.values())
        _write_smoke_report(config, result, resources, checks, pass_flag)
        _write_step13_doc(config, result, resources, pass_flag)
        print(REPORT_PATH.read_text(encoding="utf-8"))
        if not pass_flag:
            raise SystemExit(1)
    except Exception as error:
        elapsed = time.time() - start
        resources["after"] = _resource_snapshot()
        resources["elapsed_time_sec"] = elapsed
        resources["oom"] = "out of memory" in repr(error).lower()
        _failure_payload(error, resources, elapsed)
        print(REPORT_PATH.read_text(encoding="utf-8"))
        raise


if __name__ == "__main__":
    main()
