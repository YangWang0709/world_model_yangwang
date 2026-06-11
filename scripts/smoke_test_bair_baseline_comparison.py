"""End-to-end Step 12 smoke for BAIR baseline comparison."""

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

from eval.eval_bair_baseline_comparison import evaluate_baseline_comparison
from training.run_bair_baseline_comparison import run_bair_baseline_comparison
from training.run_baseline_comparison import load_yaml


CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline_comparison_bair_videomae_smoke.yaml"
REPORT_PATH = PROJECT_ROOT / "docs" / "BAIR_BASELINE_COMPARISON_SMOKE_REPORT.md"
STEP12_DOC_PATH = PROJECT_ROOT / "docs" / "STEP12_BAIR_BASELINE_COMPARISON.md"


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


def _has_shards(path: Path, pattern: str) -> bool:
    return bool(list(path.glob(pattern)))


def _ensure_inputs(config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    checks = [
        (Path(data_cfg["train_token_shard_dir"]), data_cfg.get("token_shard_glob", "tokens_shard_*.pt")),
        (Path(data_cfg["test_token_shard_dir"]), data_cfg.get("token_shard_glob", "tokens_shard_*.pt")),
        (Path(data_cfg["train_importance_shard_dir"]), data_cfg.get("importance_shard_glob", "importance_shard_*.pt")),
        (Path(data_cfg["test_importance_shard_dir"]), data_cfg.get("importance_shard_glob", "importance_shard_*.pt")),
    ]
    for shard_dir, pattern in checks:
        if not _has_shards(shard_dir, pattern):
            raise FileNotFoundError(f"Missing BAIR shards under {shard_dir} with pattern {pattern}")
    if not Path(config["learned_selector"]["checkpoint"]).exists():
        raise FileNotFoundError("Missing Step 11E selector checkpoint; run Step 11E first.")
    if not Path(config["teacher_reference"]["checkpoint"]).exists():
        raise FileNotFoundError("Missing Step 11C teacher checkpoint; run Step 11C first.")


def _clean_run_dir(config: dict[str, Any]) -> Path:
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    expected = (PROJECT_ROOT / "runs" / "baseline_comparison_bair_videomae_smoke_v1").resolve()
    resolved = run_dir.resolve()
    if resolved != expected:
        raise ValueError(f"Refusing to clean unexpected run dir: {run_dir}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return run_dir


def _row(rows: list[dict[str, Any]], policy: str, seed: int | None = None) -> dict[str, Any] | None:
    for item in rows:
        if item.get("policy") == policy and (seed is None or int(item.get("seed", -1)) == int(seed)):
            return item
    return None


def _write_report(
    config: dict[str, Any],
    result: dict[str, Any],
    resources: dict[str, Any],
    checks: dict[str, bool],
    pass_flag: bool,
) -> None:
    command = (
        "/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab "
        "python scripts/smoke_test_bair_baseline_comparison.py"
    )
    summary_md = Path(result["summary_md"]).read_text(encoding="utf-8")
    aggregate = result["aggregate"]
    report = [
        "# BAIR Baseline Comparison Smoke Report",
        "",
        f"Command: `{command}`",
        "",
        f"- run dir: `{result['run_dir']}`",
        f"- policies: `{', '.join(result['policies'])}`",
        f"- summary_json: `{result['summary_json']}`",
        f"- summary_csv: `{result['summary_csv']}`",
        f"- summary_md: `{result['summary_md']}`",
        "",
        "## Baseline Summary Table",
        "",
        summary_md,
        "",
        "## Learned vs Random",
        "",
        "```json",
        json.dumps(aggregate.get("learned_vs_random", {}), indent=2),
        "```",
        "",
        "## Learned vs Teacher-Importance TopK",
        "",
        "```json",
        json.dumps(aggregate.get("learned_vs_oracle", {}), indent=2),
        "```",
        "",
        "## Sanity Gate",
        "",
        "```json",
        json.dumps(aggregate.get("sanity_gate", {}), indent=2),
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
        f"BAIR_BASELINE_COMPARISON_SMOKE_PASS = {str(pass_flag).lower()}",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_step12_doc(
    config: dict[str, Any],
    result: dict[str, Any],
    resources: dict[str, Any],
    pass_flag: bool,
    pytest_result: str = "Run after Step 12 smoke; final result is recorded in the final response/local summary.",
) -> None:
    summary_md = Path(result["summary_md"]).read_text(encoding="utf-8")
    aggregate = result["aggregate"]
    doc = [
        "# STEP12 BAIR Baseline Comparison Report",
        "",
        "## 1. Goal",
        "",
        "Step 12 compares token selection policies on BAIR Robot Pushing small represented as VideoMAE token shards.",
        "",
        "## 2. Cloud Server Decision",
        "",
        "No cloud server is required for this stage because the run reads existing token and importance shards and does not run the VideoMAE encoder.",
        "",
        f"- OOM: `{resources.get('oom', False)}`",
        f"- max RAM used GiB: `{resources.get('max_ram_used_gib')}`",
        f"- max GPU memory used GiB: `{resources.get('max_gpu_mem_used_gib')}`",
        f"- cloud recommendation: `{resources.get('cloud_recommendation')}`",
        "",
        "## 3. Why Baselines Are Needed",
        "",
        "Step 11F proved the learned-selector compressed Student pipeline runs end to end. Step 12 checks whether the learned selector compares favorably to Random-K, Uniform-K, and an oracle-like Teacher-Importance TopK baseline at the same token budget.",
        "",
        "## 4. Baselines",
        "",
        "- Full-token Teacher reference",
        "- Random-K Student",
        "- Uniform-K Student",
        "- Teacher-Importance TopK Student",
        "- Learned Selector Student",
        "",
        "## 5. Training Protocol",
        "",
        f"- topK: `{config['selection']['topk']}`",
        "- token retention ratio: `16 / 392 = 0.04081632653061224`",
        f"- max_steps per baseline: `{config['training']['max_steps']}`",
        f"- batch_size: `{config['training']['batch_size']}`",
        "- all Student baselines train independent `TokenCompressor + StudentWorldModel` modules",
        "- Teacher and learned selector remain frozen",
        "",
        "## 6. Metrics",
        "",
        "- student_future_mse",
        "- teacher_mse",
        "- student_teacher_ratio",
        "- token_retention_ratio",
        "- selector_target_topK_overlap",
        "- selected_teacher_importance_mean",
        "- selected_vs_random_importance_gap",
        "",
        "## 7. Results Table",
        "",
        summary_md,
        "",
        "## 8. Sanity Gate",
        "",
        "```json",
        json.dumps(aggregate.get("sanity_gate", {}), indent=2),
        "```",
        "",
        "## 9. Pytest Result",
        "",
        pytest_result,
        "",
        "## 10. Git Commit",
        "",
        "Branch: `feature/tgpawb-step3-token-extraction`. The final commit hash and push status are recorded after commit/push in `STEP12_LOCAL_SUMMARY.md` and the final response. No PR is created.",
        "",
        "## 11. What Was Not Done",
        "",
        "- no new model download",
        "- no new dataset download",
        "- no BAIR re-download",
        "- no VideoMAE training",
        "- no Teacher retraining",
        "- no selector retraining",
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
        "## 12. Next Step Recommendation",
        "",
        "Step 13 should choose the next direction from the baseline result: expand BAIR if learned selection is stable, improve selector supervision if it is not, or add task/action conditioning if the paper direction needs stronger task grounding.",
    ]
    STEP12_DOC_PATH.write_text("\n".join(doc) + "\n", encoding="utf-8")


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
        "# BAIR Baseline Comparison Smoke Report\n\n"
        "Step 12 failed before producing a complete smoke report.\n\n"
        "```json\n"
        + json.dumps(payload, indent=2)
        + "\n```\n\n"
        "BAIR_BASELINE_COMPARISON_SMOKE_PASS = false\n",
        encoding="utf-8",
    )


def main() -> None:
    start = time.time()
    before = _resource_snapshot()
    config = load_yaml(CONFIG_PATH)
    resources: dict[str, Any] = {"before": before}
    try:
        _ensure_inputs(config)
        run_dir = _clean_run_dir(config)
        result = run_bair_baseline_comparison(config)
        eval_result = evaluate_baseline_comparison(run_dir)
        result.update(
            {
                "summary_json": eval_result["summary_json"],
                "summary_csv": eval_result["summary_csv"],
                "summary_md": eval_result["summary_md"],
                "aggregate": eval_result["aggregate"],
            }
        )
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
            "consider 4090 / 48GB for larger-scale BAIR baseline runs"
            if resources["max_ram_used_gib"] >= max_ram_gb_warning or gpu_fraction >= max_gpu_mem_fraction_warning
            else "not required for Step 12 smoke"
        )

        rows = result["rows"]
        learned = _row(rows, "learned_selector")
        oracle = _row(rows, "teacher_importance_topk")
        policies = {row["policy"] for row in rows}
        required_policies = {"random_k", "uniform_k", "teacher_importance_topk", "learned_selector"}
        checks = {
            "baseline_summary_json_exists": Path(result["summary_json"]).exists(),
            "baseline_summary_csv_exists": Path(result["summary_csv"]).exists(),
            "baseline_summary_md_exists": Path(result["summary_md"]).exists(),
            "required_policies_present": required_policies.issubset(policies),
            "learned_selector_ratio_finite": learned is not None and _finite(learned.get("student_teacher_ratio")),
            "learned_selector_importance_finite": (
                learned is not None and _finite(learned.get("selected_teacher_importance_mean"))
            ),
            "teacher_importance_topk_importance_finite": (
                oracle is not None and _finite(oracle.get("selected_teacher_importance_mean"))
            ),
            "sanity_gate_pass": bool(result["aggregate"]["sanity_gate"]["pass"]),
            "ram_below_warning": resources["max_ram_used_gib"] < max_ram_gb_warning,
        }
        pass_flag = all(checks.values())
        _write_report(config, result, resources, checks, pass_flag)
        _write_step12_doc(config, result, resources, pass_flag)
        print(REPORT_PATH.read_text(encoding="utf-8"))
        if not pass_flag:
            raise SystemExit(1)
    except Exception as error:
        elapsed = time.time() - start
        resources["after"] = _resource_snapshot()
        resources["elapsed_time_sec"] = elapsed
        _failure_payload(error, resources, elapsed)
        print(REPORT_PATH.read_text(encoding="utf-8"))
        raise


if __name__ == "__main__":
    main()
