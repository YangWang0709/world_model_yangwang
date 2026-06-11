"""Run Step17 BAIR context bottleneck validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bair_context_bottleneck_validation import load_yaml, write_context_bottleneck_summaries
from scripts.extract_context_tokens import extract_context_tokens_from_config
from scripts.generate_context_predictive_importance import generate_context_predictive_importance
from scripts.run_bair_500_64_scale_validation import ensure_resource_limits, resource_snapshot
from training.train_context_bottleneck_world_model import run_context_bottleneck_baselines, train_context_bottleneck_world_model
from training.train_context_teacher import run_context_teacher_training
from training.train_unified_context_selector import run_unified_context_selector_training


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "context_bottleneck_bair_1000_128.yaml"
PROTECTED_MARKERS = (
    "bair_videomae_smoke",
    "bair_500_64",
    "bair_1000_128_multiseed_validation_v1",
    "teacher_bair_videomae_1000_128_v1",
    "student_selector_bair_videomae_1000_128_weighted_mse_multiseed_v1",
    "student_world_model_bair_videomae_1000_128_weighted_mse_multiseed_v1",
    "downstream_utilization_bair_1000_128_v1",
)


@dataclass(frozen=True)
class StepDefinition:
    name: str
    description: str


STEP_DEFINITIONS = [
    StepDefinition("resource_check", "check local disk/RAM/GPU and fixed inputs"),
    StepDefinition("export_context_windows", "export 16-frame BAIR context/current/future windows from existing TFDS"),
    StepDefinition("extract_context_tokens", "extract context/current/future frozen VideoMAE tokens"),
    StepDefinition("train_context_teacher", "train full-context ContextTeacher"),
    StepDefinition("generate_context_importance", "generate context-only predictive importance"),
    StepDefinition("train_unified_context_selector", "train UnifiedPredictiveImportanceSelector in context mode"),
    StepDefinition("train_context_bottleneck_world_model", "train learned context bottleneck world model"),
    StepDefinition("baseline_comparison", "train current/random/uniform/oracle/learned/hybrid baselines"),
    StepDefinition("write_summary", "write Step17 context bottleneck summaries"),
]


def step_names() -> list[str]:
    return [step.name for step in STEP_DEFINITIONS]


def is_step17_owned_output_path(path_value: str | Path) -> bool:
    value = str(path_value)
    if any(marker in value for marker in PROTECTED_MARKERS):
        return False
    return any(
        marker in value
        for marker in (
            "bair_context_windows_1000_128",
            "context_token_shards",
            "context_importance_shards",
            "context_teacher_bair_1000_128",
            "unified_context_selector_bair_1000_128",
            "context_bottleneck_world_model_bair_1000_128",
            "context_bottleneck_baseline_bair_1000_128",
            "context_bottleneck_bair_1000_128",
        )
    )


def validate_step17_output_paths(config: dict[str, Any]) -> None:
    values = [
        config["dataset"]["output_dir"],
        config["tokens"]["output_root"],
        config["importance"]["output_root"],
        str(Path(config["output"]["run_root"]) / config["output"]["run_name"]),
        str(Path(config["output"]["run_root"]) / config["teacher"]["run_name"]),
        str(Path(config["output"]["run_root"]) / config["unified_selector"]["run_name"]),
        str(Path(config["output"]["run_root"]) / config["context_bottleneck_world_model"]["run_name"]),
        str(Path(config["output"]["run_root"]) / config["baselines"]["run_name"]),
    ]
    bad = [value for value in values if not is_step17_owned_output_path(value)]
    if bad:
        raise ValueError(f"Refusing non-Step17 output path(s): {bad}")


def _read_json(path: str | Path, required: bool = False) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(p)
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _is_finite(value: Any) -> bool:
    try:
        return torch.isfinite(torch.tensor(float(value))).item()
    except (TypeError, ValueError):
        return False


def _config_path(config: dict[str, Any], key: str) -> Path:
    value = Path(config["configs"][key])
    return value if value.is_absolute() else PROJECT_ROOT / value


def write_partial_summary(
    run_dir: str | Path,
    *,
    current_step: str,
    completed_steps: list[str],
    status: str,
    error: str | None = None,
    resource: dict[str, Any] | None = None,
) -> Path:
    path = Path(run_dir) / "partial_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "context_bottleneck_bair_1000_128",
        "status": status,
        "current_step": current_step,
        "completed_steps": completed_steps,
        "error": error,
        "resource": resource or {},
        "updated_at_unix": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _resource_delta(before: dict[str, Any], after: dict[str, Any], elapsed: float, oom: bool) -> dict[str, Any]:
    gpu_total = after.get("gpu_mem_total_gib") or 0.0
    gpu_used = after.get("gpu_mem_used_gib") or 0.0
    return {
        "before": before,
        "after": after,
        "elapsed_time_sec": round(elapsed, 3),
        "max_ram_used_gib": after.get("ram_used_gib"),
        "max_gpu_mem_used_gib": after.get("gpu_mem_used_gib"),
        "gpu_mem_fraction": float(gpu_used) / float(gpu_total) if gpu_total else None,
        "oom": bool(oom),
        "cloud_recommendation": "recommended only after lowering batch_size or token_chunk_size still fails" if oom else "not required for Step 17 local context pipeline validation",
    }


def _context_windows_complete(config: dict[str, Any]) -> bool:
    summary = _read_json(Path(config["dataset"]["output_dir"]) / "export_summary.json")
    return (
        bool(summary.get("export_success"))
        and int(summary.get("splits", {}).get("train", {}).get("exported", 0)) == int(config["dataset"]["train_samples"])
        and int(summary.get("splits", {}).get("test", {}).get("exported", 0)) == int(config["dataset"]["test_samples"])
    )


def _tokens_complete(config: dict[str, Any]) -> bool:
    summary = _read_json(Path(config["tokens"]["output_root"]) / "extraction_summary.json")
    train = summary.get("split_summaries", {}).get("train", {})
    test = summary.get("split_summaries", {}).get("test", {})
    return (
        not bool(summary.get("used_fallback", True))
        and int(train.get("dataset_size", 0)) == int(config["dataset"]["train_samples"])
        and int(test.get("dataset_size", 0)) == int(config["dataset"]["test_samples"])
        and train.get("context_token_shape") is not None
    )


def _teacher_complete(config: dict[str, Any]) -> bool:
    summary = _read_json(Path(config["output"]["run_root"]) / config["teacher"]["run_name"] / "summary.json")
    return Path(config["teacher"]["checkpoint"]).exists() and int(summary.get("num_steps", 0)) == int(config["teacher"]["max_steps"]) and _is_finite(summary.get("eval_mse"))


def _importance_complete(config: dict[str, Any]) -> bool:
    summary = _read_json(Path(config["importance"]["output_root"]) / "importance_summary.json")
    return int(summary.get("train_num_samples", 0)) == int(config["dataset"]["train_samples"]) and int(summary.get("test_num_samples", 0)) == int(config["dataset"]["test_samples"])


def _selector_complete(config: dict[str, Any]) -> bool:
    summary = _read_json(Path(config["output"]["run_root"]) / config["unified_selector"]["run_name"] / "summary.json")
    return Path(config["unified_selector"]["checkpoint"]).exists() and summary.get("trained_mode") == "context" and not bool(summary.get("trained_current_importance", True))


def _world_model_complete(config: dict[str, Any]) -> bool:
    summary = _read_json(Path(config["output"]["run_root"]) / config["context_bottleneck_world_model"]["run_name"] / "summary.json")
    return Path(config["context_bottleneck_world_model"]["checkpoint"]).exists() and _is_finite(summary.get("student_future_mse")) and not bool(summary.get("current_tokens_dropped", True))


def _baseline_complete(config: dict[str, Any]) -> bool:
    summary = _read_json(Path(config["output"]["run_root"]) / config["baselines"]["run_name"] / "baseline_aggregate.json")
    rows = summary.get("aggregate", [])
    return isinstance(rows, list) and len(rows) >= 5


COMPLETION_CHECKS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "export_context_windows": _context_windows_complete,
    "extract_context_tokens": _tokens_complete,
    "train_context_teacher": _teacher_complete,
    "generate_context_importance": _importance_complete,
    "train_unified_context_selector": _selector_complete,
    "train_context_bottleneck_world_model": _world_model_complete,
    "baseline_comparison": _baseline_complete,
}


def _run_export(config: dict[str, Any]) -> dict[str, Any]:
    conda = config["environments"]["conda_executable"]
    env = config["environments"]["tfds_export_env"]
    export_config = str(_config_path(config, "export_context_windows"))
    cmd = [conda, "run", "-n", env, "python", "scripts/export_bair_context_windows.py", "--config", export_config]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    convert_cmd = [sys.executable, "scripts/convert_bair_context_npz_to_pt.py", "--config", export_config]
    result = subprocess.run(convert_cmd, cwd=str(PROJECT_ROOT), check=False, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, convert_cmd)
    if not _context_windows_complete(config):
        raise RuntimeError("BAIR context window export incomplete")
    return _read_json(Path(config["dataset"]["output_dir"]) / "export_summary.json", required=True)


def _run_tokens(config: dict[str, Any]) -> dict[str, Any]:
    summary = extract_context_tokens_from_config(_config_path(config, "token_extraction"), overwrite=False)
    if not _tokens_complete(config):
        raise RuntimeError("BAIR context token extraction incomplete")
    return summary


def _run_teacher(config: dict[str, Any]) -> dict[str, Any]:
    summary = run_context_teacher_training(load_yaml(_config_path(config, "train_context_teacher")))
    if not _teacher_complete(config):
        raise RuntimeError("ContextTeacher incomplete")
    return summary


def _run_importance(config: dict[str, Any]) -> dict[str, Any]:
    summary = generate_context_predictive_importance(_config_path(config, "generate_context_importance"))
    if not _importance_complete(config):
        raise RuntimeError("Context importance incomplete")
    return summary


def _run_selector(config: dict[str, Any]) -> dict[str, Any]:
    summary = run_unified_context_selector_training(load_yaml(_config_path(config, "train_unified_context_selector")))
    if not _selector_complete(config):
        raise RuntimeError("Unified context selector incomplete")
    return summary


def _run_world(config: dict[str, Any]) -> dict[str, Any]:
    summary = train_context_bottleneck_world_model(load_yaml(_config_path(config, "train_context_bottleneck_world_model")))
    if not _world_model_complete(config):
        raise RuntimeError("Context bottleneck world model incomplete")
    return summary


def _run_baselines(config: dict[str, Any]) -> dict[str, Any]:
    summary = run_context_bottleneck_baselines(load_yaml(_config_path(config, "baseline_context_bottleneck")))
    if not _baseline_complete(config):
        raise RuntimeError("Context bottleneck baseline comparison incomplete")
    return summary


RUNNERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "export_context_windows": _run_export,
    "extract_context_tokens": _run_tokens,
    "train_context_teacher": _run_teacher,
    "generate_context_importance": _run_importance,
    "train_unified_context_selector": _run_selector,
    "train_context_bottleneck_world_model": _run_world,
    "baseline_comparison": _run_baselines,
}


def _required_inputs(config: dict[str, Any]) -> None:
    if not Path(config["dataset"]["tfds_data_dir"]).exists():
        raise FileNotFoundError(f"Missing existing BAIR TFDS cache: {config['dataset']['tfds_data_dir']}")
    if not Path(config["encoder"]["model_name_or_path"]).exists():
        raise FileNotFoundError(f"Missing local VideoMAE checkpoint: {config['encoder']['model_name_or_path']}")


def run_bair_context_bottleneck_validation(config_path: str | Path = DEFAULT_CONFIG, skip_completed: bool = True) -> dict[str, Any]:
    config = load_yaml(config_path)
    validate_step17_output_paths(config)
    _required_inputs(config)
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    before = resource_snapshot()
    ensure_resource_limits(config, before)
    completed: list[str] = []
    start = time.perf_counter()
    oom = False
    current_step = "resource_check"
    try:
        for step in STEP_DEFINITIONS:
            current_step = step.name
            if step.name == "resource_check":
                completed.append(step.name)
                write_partial_summary(run_dir, current_step=step.name, completed_steps=completed, status="running", resource={"before": before})
                continue
            if step.name == "write_summary":
                completed.append(step.name)
                continue
            check = COMPLETION_CHECKS[step.name]
            if skip_completed and check(config):
                print(f"SKIP_STEP17_STEP {step.name}: already complete", flush=True)
                completed.append(step.name)
                continue
            print(f"RUN_STEP17_STEP {step.name}: {step.description}", flush=True)
            RUNNERS[step.name](config)
            after_step = resource_snapshot()
            ensure_resource_limits(config, after_step)
            completed.append(step.name)
            write_partial_summary(run_dir, current_step=step.name, completed_steps=completed, status="running", resource={"before": before, "latest": after_step})
        after = resource_snapshot()
        resource = _resource_delta(before, after, time.perf_counter() - start, oom=oom)
        summary = write_context_bottleneck_summaries(config=config, resource_summary=resource)
        write_partial_summary(run_dir, current_step="complete", completed_steps=completed, status="complete", resource=resource)
        print("BAIR_CONTEXT_BOTTLENECK_VALIDATION_RUNNER_COMPLETE = true", flush=True)
        return summary
    except BaseException as exc:
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower() or "oom" in str(exc).lower()
        if oom and torch.cuda.is_available():
            torch.cuda.empty_cache()
        after = resource_snapshot()
        resource = _resource_delta(before, after, time.perf_counter() - start, oom=oom)
        write_partial_summary(run_dir, current_step=current_step, completed_steps=completed, status="failed", error=repr(exc), resource=resource)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_bair_context_bottleneck_validation(args.config, skip_completed=not args.rerun_completed)
    print("BAIR_CONTEXT_BOTTLENECK_VALIDATION_RUNNER_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
