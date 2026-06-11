"""Run Step 14 BAIR 500/64 scale validation without touching Step 11-13 outputs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
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

from eval.eval_bair_500_64_scale_validation import load_yaml, write_scale_validation_summary
from eval.eval_importance_bair_videomae_summary import evaluate_bair_importance_root
from eval.eval_student_world_model import evaluate_student_world_model
from eval.eval_teacher_prediction import evaluate_teacher
from eval.eval_teacher_student_gap import evaluate_teacher_student_gap
from scripts.extract_tokens import extract_tokens_from_config
from scripts.generate_predictive_importance import generate_predictive_importance
from training.baseline_student_world_model_trainer import train_baseline_comparison
from training.selector_ablation_trainer import SelectorVariantTrainer
from training.student_world_model_trainer import run_student_world_model_training
from training.teacher_trainer import run_teacher_training


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bair_500_64_scale_validation.yaml"
PROTECTED_OUTPUT_MARKERS = (
    "bair_videomae_smoke",
    "bair_videomae_teacher_smoke",
    "teacher_bair_videomae_smoke_v1",
    "student_selector_bair_videomae_smoke_v1",
    "student_world_model_bair_videomae_smoke_v1",
    "baseline_comparison_bair_videomae_smoke_v1",
    "selector_ablation_bair_videomae_smoke_v1",
)


@dataclass(frozen=True)
class StepDefinition:
    name: str
    description: str


STEP_DEFINITIONS = [
    StepDefinition("resource_check", "check local disk/RAM/GPU and required local inputs"),
    StepDefinition("export_subset", "export BAIR 500/64 subset from existing TFDS"),
    StepDefinition("extract_tokens", "extract frozen VideoMAE tokens for BAIR 500/64"),
    StepDefinition("train_teacher", "train full-token TeacherWorldModel"),
    StepDefinition("generate_importance", "generate Teacher predictive importance"),
    StepDefinition("train_selector", "train weighted_mse_alpha2 selector"),
    StepDefinition("train_student_world_model", "train selector-compressed StudentWorldModel"),
    StepDefinition("baseline_comparison", "train random/uniform/oracle/learned baselines"),
    StepDefinition("write_summary", "write Step 14 scale validation summary"),
]


def step_names() -> list[str]:
    return [step.name for step in STEP_DEFINITIONS]


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: str | Path, required: bool = True) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        if required:
            raise FileNotFoundError(json_path)
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def _count_jsonl(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _bytes_to_gib(value: int | float) -> float:
    return round(float(value) / (1024**3), 3)


def _ram_snapshot() -> dict[str, float | None]:
    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        return {
            "ram_total_gib": _bytes_to_gib(memory.total),
            "ram_available_gib": _bytes_to_gib(memory.available),
            "ram_used_gib": _bytes_to_gib(memory.used),
        }
    except Exception:
        meminfo: dict[str, int] = {}
        path = Path("/proc/meminfo")
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                key, value = line.split(":", 1)
                meminfo[key] = int(value.strip().split()[0]) * 1024
        total = meminfo.get("MemTotal", 0)
        available = meminfo.get("MemAvailable", 0)
        return {
            "ram_total_gib": _bytes_to_gib(total),
            "ram_available_gib": _bytes_to_gib(available),
            "ram_used_gib": _bytes_to_gib(total - available),
        }


def _gpu_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch_cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_mem_total_gib": None,
        "gpu_mem_used_gib": None,
        "gpu_mem_free_gib": None,
    }
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device("cuda"))
        payload.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_mem_total_gib": _bytes_to_gib(total_bytes),
                "gpu_mem_free_gib": _bytes_to_gib(free_bytes),
                "gpu_mem_used_gib": _bytes_to_gib(total_bytes - free_bytes),
            }
        )
    return payload


def resource_snapshot() -> dict[str, Any]:
    disk = shutil.disk_usage(PROJECT_ROOT)
    return {
        "disk_total_gib": _bytes_to_gib(disk.total),
        "disk_used_gib": _bytes_to_gib(disk.used),
        "disk_free_gib": _bytes_to_gib(disk.free),
        **_ram_snapshot(),
        **_gpu_snapshot(),
    }


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
        "cloud_recommendation": (
            "recommended only after lowering batch_size or token_chunk_size still fails"
            if oom
            else "not required for Step 14 local 500/64 validation"
        ),
    }


def ensure_resource_limits(config: dict[str, Any], snapshot: dict[str, Any]) -> None:
    limits = config.get("resource_limits", {})
    min_disk = float(limits.get("min_free_disk_gb", 0.0) or 0.0)
    max_ram = float(limits.get("max_ram_gb_warning", 0.0) or 0.0)
    if min_disk and float(snapshot.get("disk_free_gib", 0.0)) < min_disk:
        raise RuntimeError(f"Free disk is below {min_disk} GiB: {snapshot.get('disk_free_gib')} GiB")
    if max_ram and float(snapshot.get("ram_used_gib", 0.0)) > max_ram:
        raise RuntimeError(f"RAM usage exceeded {max_ram} GiB: {snapshot.get('ram_used_gib')} GiB")


def is_step14_owned_output_path(path_value: str | Path) -> bool:
    value = str(path_value)
    if any(marker in value for marker in PROTECTED_OUTPUT_MARKERS):
        return False
    return "500_64" in value or "500/64" in value or "bair_500_64_scale_validation" in value


def validate_step14_output_paths(config: dict[str, Any]) -> None:
    output_values = [
        config["dataset"]["subset_dir"],
        config["tokens"]["output_root"],
        config["importance"]["output_root"],
        str(Path(config["output"]["run_root"]) / config["output"]["run_name"]),
        str(Path(config["output"]["run_root"]) / config["teacher"]["run_name"]),
        str(Path(config["output"]["run_root"]) / config["selector"]["run_name"]),
        str(Path(config["output"]["run_root"]) / config["student_world_model"]["run_name"]),
        str(Path(config["output"]["run_root"]) / config["baseline"]["run_name"]),
    ]
    bad = [value for value in output_values if not is_step14_owned_output_path(value)]
    if bad:
        raise ValueError(f"Refusing non-Step14 output paths: {bad}")


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
        "stage": "bair_500_64_scale_validation",
        "status": status,
        "current_step": current_step,
        "completed_steps": completed_steps,
        "error": error,
        "resource": resource or {},
        "updated_at_unix": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _run_dir(config: dict[str, Any], run_name: str) -> Path:
    return Path(config["output"]["run_root"]) / run_name


def _checkpoint(run_dir: Path, prefix: str, steps: int) -> Path:
    return run_dir / "checkpoints" / f"{prefix}_step_{steps:06d}.pt"


def _subset_complete(config: dict[str, Any]) -> bool:
    root = Path(config["dataset"]["subset_dir"])
    summary = _read_json(root / "export_summary.json", required=False)
    try:
        train_count = _count_jsonl(root / "train" / "metadata.jsonl")
        test_count = _count_jsonl(root / "test" / "metadata.jsonl")
    except FileNotFoundError:
        return False
    return bool(summary.get("export_success")) and train_count == int(config["dataset"]["train_samples"]) and test_count == int(config["dataset"]["test_samples"])


def _tokens_complete(config: dict[str, Any]) -> bool:
    root = Path(config["tokens"]["output_root"])
    summary = _read_json(root / "extraction_summary.json", required=False)
    train = summary.get("split_summaries", {}).get("train", {})
    test = summary.get("split_summaries", {}).get("test", {})
    shape = train.get("output_token_shape")
    return (
        bool(summary)
        and not bool(summary.get("used_fallback", True))
        and int(train.get("dataset_size", 0)) == int(config["dataset"]["train_samples"])
        and int(test.get("dataset_size", 0)) == int(config["dataset"]["test_samples"])
        and isinstance(shape, list)
        and shape[1:] == [392, 768]
    )


def _teacher_complete(config: dict[str, Any]) -> bool:
    run_dir = _run_dir(config, config["teacher"]["run_name"])
    checkpoint = Path(config["teacher"]["checkpoint"])
    eval_summary = _read_json(run_dir / "eval_summary.json", required=False)
    train_summary = _read_json(run_dir / "summary.json", required=False)
    return checkpoint.exists() and _is_finite(eval_summary.get("eval_mse")) and int(train_summary.get("num_steps", 0)) == int(config["teacher"]["max_steps"])


def _importance_complete(config: dict[str, Any]) -> bool:
    root = Path(config["importance"]["output_root"])
    summary = _read_json(root / "importance_summary.json", required=False)
    eval_summary = _read_json(root / "eval_importance_summary.json", required=False)
    return (
        int(summary.get("train_num_samples", 0)) == int(config["dataset"]["train_samples"])
        and int(summary.get("test_num_samples", 0)) == int(config["dataset"]["test_samples"])
        and int(eval_summary.get("num_samples", 0)) == int(config["dataset"]["train_samples"]) + int(config["dataset"]["test_samples"])
        and _is_finite(eval_summary.get("importance_mean"))
    )


def _selector_complete(config: dict[str, Any]) -> bool:
    run_dir = _run_dir(config, config["selector"]["run_name"])
    checkpoint = Path(config["selector"]["checkpoint"])
    summary = _read_json(run_dir / "summary.json", required=False)
    return checkpoint.exists() and int(summary.get("num_steps", 0)) == int(config["selector"]["max_steps"]) and _is_finite(summary.get("test_importance_mse"))


def _student_complete(config: dict[str, Any]) -> bool:
    run_dir = _run_dir(config, config["student_world_model"]["run_name"])
    checkpoint = Path(config["student_world_model"]["checkpoint"])
    summary = _read_json(run_dir / "summary.json", required=False)
    eval_summary = _read_json(run_dir / "eval_summary.json", required=False)
    gap = _read_json(run_dir / "teacher_student_gap_summary.json", required=False)
    return checkpoint.exists() and int(summary.get("num_steps", 0)) == int(config["student_world_model"]["max_steps"]) and _is_finite(eval_summary.get("student_future_mse")) and _is_finite(gap.get("student_teacher_ratio"))


def _baseline_complete(config: dict[str, Any]) -> bool:
    run_dir = _run_dir(config, config["baseline"]["run_name"])
    summary = _read_json(run_dir / "baseline_summary.json", required=False)
    rows = summary.get("rows", [])
    return isinstance(rows, list) and len(rows) >= 6


COMPLETION_CHECKS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "export_subset": _subset_complete,
    "extract_tokens": _tokens_complete,
    "train_teacher": _teacher_complete,
    "generate_importance": _importance_complete,
    "train_selector": _selector_complete,
    "train_student_world_model": _student_complete,
    "baseline_comparison": _baseline_complete,
}


def _run_subprocess(command: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    print("RUN", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(cwd), check=False, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def _config_path(config: dict[str, Any], key: str) -> Path:
    return _resolve(config["configs"][key])


def _run_export_subset(config: dict[str, Any]) -> dict[str, Any]:
    conda = config["environments"]["conda_executable"]
    env = config["environments"]["tfds_export_env"]
    export_config = str(_config_path(config, "export_subset"))
    _run_subprocess([conda, "run", "-n", env, "python", "scripts/export_bair_subset_npz.py", "--config", export_config])
    _run_subprocess([sys.executable, "scripts/convert_bair_npz_subset_to_pt.py", "--config", export_config])
    summary = _read_json(Path(config["dataset"]["subset_dir"]) / "export_summary.json")
    if not _subset_complete(config):
        raise RuntimeError(f"BAIR 500/64 subset export incomplete: {summary}")
    return summary


def _run_token_extraction(config: dict[str, Any]) -> dict[str, Any]:
    summary = extract_tokens_from_config(_config_path(config, "token_extraction"), overwrite=True)
    if not _tokens_complete(config):
        raise RuntimeError("BAIR 500/64 VideoMAE token extraction did not produce expected [B,392,768] shards")
    return summary


def _run_teacher(config: dict[str, Any]) -> dict[str, Any]:
    teacher_cfg = load_yaml(_config_path(config, "train_teacher"))
    train_summary = run_teacher_training(teacher_cfg)
    evaluate_teacher(teacher_cfg, train_summary["checkpoint_path"])
    if not _teacher_complete(config):
        raise RuntimeError("BAIR 500/64 Teacher training/eval incomplete")
    return train_summary


def _run_importance(config: dict[str, Any]) -> dict[str, Any]:
    importance_cfg = load_yaml(_config_path(config, "generate_importance"))
    summary = generate_predictive_importance(importance_cfg)
    evaluate_bair_importance_root(config["importance"]["output_root"])
    if not _importance_complete(config):
        raise RuntimeError("BAIR 500/64 predictive importance incomplete")
    return summary


def _run_selector(config: dict[str, Any]) -> dict[str, Any]:
    selector_cfg = load_yaml(_config_path(config, "train_selector"))
    variant = selector_cfg["loss_variants"][0]
    run_dir = Path(selector_cfg["output"]["run_root"]) / selector_cfg["output"]["run_name"]
    summary = SelectorVariantTrainer(selector_cfg, variant, run_dir).train()
    if not _selector_complete(config):
        raise RuntimeError("BAIR 500/64 weighted_mse_alpha2 selector training incomplete")
    return summary


def _run_student(config: dict[str, Any]) -> dict[str, Any]:
    student_cfg = load_yaml(_config_path(config, "train_student_world_model"))
    summary = run_student_world_model_training(student_cfg)
    checkpoint = summary["checkpoint_path"]
    evaluate_student_world_model(student_cfg, checkpoint, split="test")
    evaluate_teacher_student_gap(
        student_cfg,
        student_checkpoint_path=checkpoint,
        teacher_checkpoint_path=student_cfg["teacher_reference"]["checkpoint"],
        split="test",
    )
    if not _student_complete(config):
        raise RuntimeError("BAIR 500/64 StudentWorldModel training/eval incomplete")
    return summary


def _run_baseline(config: dict[str, Any]) -> dict[str, Any]:
    baseline_cfg = load_yaml(_config_path(config, "baseline_comparison"))
    summary = train_baseline_comparison(baseline_cfg)
    if not _baseline_complete(config):
        raise RuntimeError("BAIR 500/64 baseline comparison incomplete")
    return summary


RUNNERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "export_subset": _run_export_subset,
    "extract_tokens": _run_token_extraction,
    "train_teacher": _run_teacher,
    "generate_importance": _run_importance,
    "train_selector": _run_selector,
    "train_student_world_model": _run_student,
    "baseline_comparison": _run_baseline,
}


def _required_inputs(config: dict[str, Any]) -> None:
    tfds_dir = Path(config["dataset"]["tfds_data_dir"])
    model_path = Path(config["encoder"]["model_name_or_path"])
    if not tfds_dir.exists():
        raise FileNotFoundError(f"Missing BAIR TFDS dir {tfds_dir}; run Step 11A-fix first, without re-downloading here.")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing local VideoMAE checkpoint {model_path}; do not download in Step 14.")


def run_scale_validation(config_path: str | Path = DEFAULT_CONFIG, skip_completed: bool = True) -> dict[str, Any]:
    config = load_yaml(config_path)
    validate_step14_output_paths(config)
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    before = resource_snapshot()
    ensure_resource_limits(config, before)
    _required_inputs(config)

    start = time.perf_counter()
    completed: list[str] = []
    oom = False
    try:
        for step in STEP_DEFINITIONS:
            if step.name in {"resource_check", "write_summary"}:
                completed.append(step.name)
                write_partial_summary(run_dir, current_step=step.name, completed_steps=completed, status="running", resource={"before": before})
                continue
            check = COMPLETION_CHECKS[step.name]
            if skip_completed and check(config):
                print(f"SKIP_STEP14_STEP {step.name}: already complete", flush=True)
                completed.append(step.name)
                continue
            print(f"RUN_STEP14_STEP {step.name}: {step.description}", flush=True)
            RUNNERS[step.name](config)
            after_step = resource_snapshot()
            ensure_resource_limits(config, after_step)
            completed.append(step.name)
            write_partial_summary(
                run_dir,
                current_step=step.name,
                completed_steps=completed,
                status="running",
                resource={"before": before, "latest": after_step},
            )
        after = resource_snapshot()
        resource = _resource_delta(before, after, time.perf_counter() - start, oom=oom)
        summary = write_scale_validation_summary(config, resource_summary=resource)
        write_partial_summary(run_dir, current_step="complete", completed_steps=completed, status="complete", resource=resource)
        print("BAIR_500_64_SCALE_VALIDATION_RUNNER_COMPLETE = true", flush=True)
        return summary
    except BaseException as exc:
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower() or "oom" in str(exc).lower()
        if oom and torch.cuda.is_available():
            torch.cuda.empty_cache()
        after = resource_snapshot()
        resource = _resource_delta(before, after, time.perf_counter() - start, oom=oom)
        write_partial_summary(
            run_dir,
            current_step=completed[-1] if completed else "resource_check",
            completed_steps=completed,
            status="failed",
            error=repr(exc),
            resource=resource,
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_scale_validation(args.config, skip_completed=not args.rerun_completed)
    print("BAIR_500_64_SCALE_VALIDATION_RUNNER_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
