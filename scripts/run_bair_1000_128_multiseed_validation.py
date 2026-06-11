"""Run Step 15 BAIR 1000/128 multi-seed validation without touching earlier outputs."""

from __future__ import annotations

import copy
import argparse
import gc
import json
import math
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

from eval.eval_bair_1000_128_multiseed_validation import (
    load_yaml,
    write_baseline_multiseed_aggregate,
    write_multiseed_validation_summary,
    write_selector_multiseed_summary,
    write_student_multiseed_summary,
)
from eval.eval_importance_bair_videomae_summary import evaluate_bair_importance_root
from eval.eval_student_world_model import evaluate_student_world_model
from eval.eval_teacher_prediction import evaluate_teacher
from eval.eval_teacher_student_gap import evaluate_teacher_student_gap
from scripts.extract_tokens import extract_tokens_from_config
from scripts.generate_predictive_importance import generate_predictive_importance
from scripts.run_bair_500_64_scale_validation import ensure_resource_limits, resource_snapshot
from training.baseline_student_world_model_trainer import BaselineStudentWorldModelTrainer, write_baseline_summaries
from training.selector_ablation_trainer import SelectorVariantTrainer
from training.student_world_model_trainer import run_student_world_model_training
from training.teacher_trainer import run_teacher_training


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bair_1000_128_multiseed_validation.yaml"
PROTECTED_OUTPUT_MARKERS = (
    "bair_videomae_smoke",
    "bair_videomae_teacher_smoke",
    "teacher_bair_videomae_smoke_v1",
    "student_selector_bair_videomae_smoke_v1",
    "student_world_model_bair_videomae_smoke_v1",
    "baseline_comparison_bair_videomae_smoke_v1",
    "selector_ablation_bair_videomae_smoke_v1",
    "bair_videomae_500_64",
    "bair_videomae_teacher_500_64",
    "bair_robot_pushing_small_subset_500_64",
    "teacher_bair_videomae_500_64_v1",
    "student_selector_bair_videomae_500_64_weighted_mse_v1",
    "student_world_model_bair_videomae_500_64_weighted_mse_v1",
    "baseline_comparison_bair_videomae_500_64_v1",
    "bair_500_64_scale_validation_v1",
)


@dataclass(frozen=True)
class StepDefinition:
    name: str
    description: str


STEP_DEFINITIONS = [
    StepDefinition("resource_check", "check local disk/RAM/GPU and required local inputs"),
    StepDefinition("export_subset", "export BAIR 1000/128 subset from existing TFDS"),
    StepDefinition("extract_tokens", "extract frozen VideoMAE tokens for BAIR 1000/128"),
    StepDefinition("train_teacher", "train full-token TeacherWorldModel"),
    StepDefinition("generate_importance", "generate Teacher predictive importance"),
    StepDefinition("train_selectors", "train weighted_mse_alpha2 selectors for seeds 0/1/2"),
    StepDefinition("train_student_world_models", "train selector-compressed StudentWorldModel for seeds 0/1/2"),
    StepDefinition("baseline_comparison", "train random/uniform/oracle/learned multi-seed baselines"),
    StepDefinition("write_summary", "write Step 15 multi-seed validation summary"),
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
            else "not required for Step 15 local 1000/128 multi-seed validation"
        ),
    }


def is_step15_owned_output_path(path_value: str | Path) -> bool:
    value = str(path_value)
    if any(marker in value for marker in PROTECTED_OUTPUT_MARKERS):
        return False
    return "1000_128" in value or "1000/128" in value or "bair_1000_128_multiseed_validation" in value


def validate_step15_output_paths(config: dict[str, Any]) -> None:
    output_values = [
        config["dataset"]["subset_dir"],
        config["tokens"]["output_root"],
        config["importance"]["output_root"],
        str(Path(config["output"]["run_root"]) / config["output"]["run_name"]),
        str(Path(config["output"]["run_root"]) / config["teacher"]["run_name"]),
        str(Path(config["output"]["run_root"]) / config["selector"]["run_root_name"]),
        str(Path(config["output"]["run_root"]) / config["student_world_model"]["run_root_name"]),
        str(Path(config["output"]["run_root"]) / config["baseline"]["run_name"]),
    ]
    bad = [value for value in output_values if not is_step15_owned_output_path(value)]
    if bad:
        raise ValueError(f"Refusing non-Step15 output paths: {bad}")


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
        "stage": "bair_1000_128_multiseed_validation",
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


def selector_seed_run_name(seed: int) -> str:
    return f"weighted_mse_alpha2_seed{int(seed)}"


def student_seed_run_name(seed: int) -> str:
    return f"learned_selector_weighted_mse_alpha2_seed{int(seed)}"


def selector_checkpoint_path(config: dict[str, Any], seed: int) -> Path:
    root = _run_dir(config, config["selector"]["run_root_name"])
    return root / selector_seed_run_name(seed) / "checkpoints" / f"student_selector_step_{int(config['selector']['max_steps']):06d}.pt"


def student_checkpoint_path(config: dict[str, Any], seed: int) -> Path:
    root = _run_dir(config, config["student_world_model"]["run_root_name"])
    return root / student_seed_run_name(seed) / "checkpoints" / f"student_world_model_step_{int(config['student_world_model']['max_steps']):06d}.pt"


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


def _selectors_complete(config: dict[str, Any]) -> bool:
    root = _run_dir(config, config["selector"]["run_root_name"])
    summary = _read_json(root / "selector_multiseed_summary.json", required=False)
    seeds = [int(seed) for seed in config["selector"]["seeds"]]
    return (
        summary.get("aggregate", {}).get("num_seeds") == len(seeds)
        and bool(summary.get("aggregate", {}).get("sanity_gate", {}).get("pass", False))
        and all(selector_checkpoint_path(config, seed).exists() for seed in seeds)
    )


def _students_complete(config: dict[str, Any]) -> bool:
    root = _run_dir(config, config["student_world_model"]["run_root_name"])
    summary = _read_json(root / "student_world_model_multiseed_summary.json", required=False)
    seeds = [int(seed) for seed in config["student_world_model"]["learned_selector_seeds"]]
    return (
        summary.get("aggregate", {}).get("num_seeds") == len(seeds)
        and bool(summary.get("aggregate", {}).get("sanity_gate", {}).get("pass", False))
        and all(student_checkpoint_path(config, seed).exists() for seed in seeds)
    )


def _baseline_complete(config: dict[str, Any]) -> bool:
    run_dir = _run_dir(config, config["baseline"]["run_name"])
    summary = _read_json(run_dir / "baseline_summary.json", required=False)
    aggregate = _read_json(run_dir / "baseline_multiseed_aggregate.json", required=False)
    rows = summary.get("rows", [])
    return isinstance(rows, list) and len(rows) >= 8 and bool(aggregate.get("aggregate", {}).get("policies"))


COMPLETION_CHECKS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "export_subset": _subset_complete,
    "extract_tokens": _tokens_complete,
    "train_teacher": _teacher_complete,
    "generate_importance": _importance_complete,
    "train_selectors": _selectors_complete,
    "train_student_world_models": _students_complete,
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
        raise RuntimeError(f"BAIR 1000/128 subset export incomplete: {summary}")
    return summary


def _run_token_extraction(config: dict[str, Any]) -> dict[str, Any]:
    summary = extract_tokens_from_config(_config_path(config, "token_extraction"), overwrite=True)
    if not _tokens_complete(config):
        raise RuntimeError("BAIR 1000/128 VideoMAE token extraction did not produce expected [B,392,768] shards")
    return summary


def _run_teacher(config: dict[str, Any]) -> dict[str, Any]:
    teacher_cfg = load_yaml(_config_path(config, "train_teacher"))
    train_summary = run_teacher_training(teacher_cfg)
    evaluate_teacher(teacher_cfg, train_summary["checkpoint_path"])
    if not _teacher_complete(config):
        raise RuntimeError("BAIR 1000/128 Teacher training/eval incomplete")
    return train_summary


def _run_importance(config: dict[str, Any]) -> dict[str, Any]:
    importance_cfg = load_yaml(_config_path(config, "generate_importance"))
    summary = generate_predictive_importance(importance_cfg)
    evaluate_bair_importance_root(config["importance"]["output_root"])
    if not _importance_complete(config):
        raise RuntimeError("BAIR 1000/128 predictive importance incomplete")
    return summary


def _run_selectors(config: dict[str, Any]) -> dict[str, Any]:
    selector_cfg = load_yaml(_config_path(config, "train_selector"))
    root = _run_dir(config, config["selector"]["run_root_name"])
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant in selector_cfg.get("loss_variants", []):
        seed = int(variant.get("seed", 0))
        subrun = root / selector_seed_run_name(seed)
        print(f"RUN_STEP15_SELECTOR seed={seed} run_dir={subrun}", flush=True)
        summary = SelectorVariantTrainer(selector_cfg, variant, subrun).train()
        summary["success"] = True
        rows.append(summary)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = write_selector_multiseed_summary(rows, root)
    if not _selectors_complete(config):
        raise RuntimeError("BAIR 1000/128 selector multi-seed training incomplete")
    return result


def _student_config_for_seed(config: dict[str, Any], seed: int) -> dict[str, Any]:
    student_cfg = load_yaml(_config_path(config, "train_student_world_model"))
    student_cfg = copy.deepcopy(student_cfg)
    student_cfg["seed"] = int(config.get("seed", 42)) + int(seed)
    student_cfg["selector"]["checkpoint"] = str(selector_checkpoint_path(config, seed))
    student_cfg["output"]["run_root"] = str(_run_dir(config, config["student_world_model"]["run_root_name"]))
    student_cfg["output"]["run_name"] = student_seed_run_name(seed)
    return student_cfg


def _run_students(config: dict[str, Any]) -> dict[str, Any]:
    root = _run_dir(config, config["student_world_model"]["run_root_name"])
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in [int(value) for value in config["student_world_model"]["learned_selector_seeds"]]:
        student_cfg = _student_config_for_seed(config, seed)
        print(f"RUN_STEP15_STUDENT seed={seed} run_dir={Path(student_cfg['output']['run_root']) / student_cfg['output']['run_name']}", flush=True)
        summary = run_student_world_model_training(student_cfg)
        checkpoint = summary["checkpoint_path"]
        eval_summary = evaluate_student_world_model(student_cfg, checkpoint, split="test")
        gap_summary = evaluate_teacher_student_gap(
            student_cfg,
            student_checkpoint_path=checkpoint,
            teacher_checkpoint_path=student_cfg["teacher_reference"]["checkpoint"],
            split="test",
        )
        row = {
            **summary,
            "seed": seed,
            "policy": "learned_selector",
            "variant_name": "weighted_mse_alpha2",
            "loss_type": "weighted_mse_alpha2",
            "success": True,
            "teacher_mse": gap_summary.get("teacher_mse"),
            "student_future_mse": gap_summary.get("student_future_mse", eval_summary.get("student_future_mse")),
            "student_teacher_ratio": gap_summary.get("student_teacher_ratio"),
            "selector_target_topk_overlap": gap_summary.get("selector_target_topk_overlap"),
            "selected_teacher_importance_mean": gap_summary.get("selected_teacher_importance_mean"),
            "selected_vs_random_importance_gap": gap_summary.get("selected_vs_random_importance_gap"),
        }
        Path(summary["summary_path"]).write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = write_student_multiseed_summary(rows, root)
    if not _students_complete(config):
        raise RuntimeError("BAIR 1000/128 StudentWorldModel multi-seed training incomplete")
    return result


def _baseline_config_for_policy_seed(config: dict[str, Any], policy_name: str, seed: int) -> dict[str, Any]:
    baseline_cfg = load_yaml(_config_path(config, "baseline_comparison"))
    baseline_cfg = copy.deepcopy(baseline_cfg)
    if policy_name == "learned_selector":
        checkpoint = baseline_cfg.get("learned_selector", {}).get("checkpoints_by_seed", {}).get(seed)
        if checkpoint is None:
            checkpoint = baseline_cfg.get("learned_selector", {}).get("checkpoints_by_seed", {}).get(str(seed))
        baseline_cfg["learned_selector"]["checkpoint"] = str(checkpoint or selector_checkpoint_path(config, seed))
    return baseline_cfg


def _baseline_subrun_name(policy_name: str, seed: int) -> str:
    if policy_name == "learned_selector":
        return student_seed_run_name(seed)
    return f"{policy_name}_seed{int(seed)}"


def _run_baseline(config: dict[str, Any]) -> dict[str, Any]:
    baseline_cfg_base = load_yaml(_config_path(config, "baseline_comparison"))
    run_dir = Path(baseline_cfg_base["output"]["run_root"]) / baseline_cfg_base["output"]["run_name"]
    repeat_seeds = baseline_cfg_base["training"].get("repeat_random_seeds", {})
    rows: list[dict[str, Any]] = []
    for policy_name in baseline_cfg_base["selection"]["policies"]:
        for seed in [int(value) for value in repeat_seeds.get(policy_name, [0])]:
            baseline_cfg = _baseline_config_for_policy_seed(config, policy_name, seed)
            subrun_dir = run_dir / _baseline_subrun_name(policy_name, seed)
            print(f"RUN_STEP15_BASELINE policy={policy_name} seed={seed} run_dir={subrun_dir}", flush=True)
            trainer = BaselineStudentWorldModelTrainer(
                config=baseline_cfg,
                policy_name=policy_name,
                seed=seed,
                run_dir=subrun_dir,
            )
            row = trainer.train()
            if policy_name == "learned_selector":
                row.update(
                    {
                        "variant_name": "weighted_mse_alpha2",
                        "loss_type": "weighted_mse_alpha2",
                        "selector_checkpoint": baseline_cfg["learned_selector"]["checkpoint"],
                    }
                )
                Path(row["summary_path"]).write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary_paths = write_baseline_summaries(rows, run_dir=run_dir, output_cfg=baseline_cfg_base.get("output", {}))
    aggregate_paths = write_baseline_multiseed_aggregate(rows, run_dir=run_dir, output_cfg=baseline_cfg_base.get("output", {}))
    result = {"run_dir": str(run_dir), "rows": rows, **summary_paths, **aggregate_paths}
    if not _baseline_complete(config):
        raise RuntimeError("BAIR 1000/128 baseline multi-seed comparison incomplete")
    return result


RUNNERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "export_subset": _run_export_subset,
    "extract_tokens": _run_token_extraction,
    "train_teacher": _run_teacher,
    "generate_importance": _run_importance,
    "train_selectors": _run_selectors,
    "train_student_world_models": _run_students,
    "baseline_comparison": _run_baseline,
}


def _required_inputs(config: dict[str, Any]) -> None:
    tfds_dir = Path(config["dataset"]["tfds_data_dir"])
    model_path = Path(config["encoder"]["model_name_or_path"])
    if not tfds_dir.exists():
        raise FileNotFoundError(f"Missing BAIR TFDS dir {tfds_dir}; run Step 11A-fix first, without re-downloading here.")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing local VideoMAE checkpoint {model_path}; do not download in Step 15.")


def run_multiseed_validation(config_path: str | Path = DEFAULT_CONFIG, skip_completed: bool = True) -> dict[str, Any]:
    config = load_yaml(config_path)
    validate_step15_output_paths(config)
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    before = resource_snapshot()
    ensure_resource_limits(config, before)
    _required_inputs(config)

    start = time.perf_counter()
    completed: list[str] = []
    oom = False
    current_step = "resource_check"
    try:
        for step in STEP_DEFINITIONS:
            current_step = step.name
            if step.name in {"resource_check", "write_summary"}:
                completed.append(step.name)
                write_partial_summary(run_dir, current_step=step.name, completed_steps=completed, status="running", resource={"before": before})
                continue
            check = COMPLETION_CHECKS[step.name]
            if skip_completed and check(config):
                print(f"SKIP_STEP15_STEP {step.name}: already complete", flush=True)
                completed.append(step.name)
                continue
            print(f"RUN_STEP15_STEP {step.name}: {step.description}", flush=True)
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
        summary = write_multiseed_validation_summary(config, resource_summary=resource)
        write_partial_summary(run_dir, current_step="complete", completed_steps=completed, status="complete", resource=resource)
        print("BAIR_1000_128_MULTI_SEED_VALIDATION_RUNNER_COMPLETE = true", flush=True)
        return summary
    except BaseException as exc:
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower() or "oom" in str(exc).lower()
        if oom and torch.cuda.is_available():
            torch.cuda.empty_cache()
        after = resource_snapshot()
        resource = _resource_delta(before, after, time.perf_counter() - start, oom=oom)
        write_partial_summary(
            run_dir,
            current_step=current_step,
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
    summary = run_multiseed_validation(args.config, skip_completed=not args.rerun_completed)
    print("BAIR_1000_128_MULTI_SEED_VALIDATION_RUNNER_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
