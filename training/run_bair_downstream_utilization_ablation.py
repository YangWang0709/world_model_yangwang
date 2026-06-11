"""Run Step 16 BAIR downstream utilization ablation."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bair_downstream_utilization_ablation import (
    load_yaml,
    write_downstream_utilization_summaries,
)
from scripts.run_bair_500_64_scale_validation import ensure_resource_limits, resource_snapshot
from training.downstream_utilization_trainer import (
    selector_checkpoint_path,
    train_downstream_utilization_variant,
    write_failed_variant_summary,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "downstream_utilization_bair_1000_128.yaml"
PROTECTED_MARKERS = (
    "bair_videomae_smoke",
    "bair_videomae_teacher_smoke",
    "teacher_bair_videomae_smoke_v1",
    "student_selector_bair_videomae_smoke_v1",
    "student_world_model_bair_videomae_smoke_v1",
    "baseline_comparison_bair_videomae_smoke_v1",
    "selector_ablation_bair_videomae_smoke_v1",
    "500_64",
    "bair_500_64",
    "bair_1000_128_multiseed_validation_v1",
    "teacher_bair_videomae_1000_128_v1",
    "student_selector_bair_videomae_1000_128_weighted_mse_multiseed_v1",
    "student_world_model_bair_videomae_1000_128_weighted_mse_multiseed_v1",
    "baseline_comparison_bair_videomae_1000_128_multiseed_v1",
)


@dataclass(frozen=True)
class StepDefinition:
    name: str
    description: str


STEP_DEFINITIONS = [
    StepDefinition("resource_check", "check disk/RAM/GPU and Step15 fixed inputs"),
    StepDefinition("phase_a", "train single-seed utilization variants"),
    StepDefinition("select_phase_b", "select Phase A top-2 variants"),
    StepDefinition("phase_b", "train selected variants over seeds 0/1/2"),
    StepDefinition("write_summary", "write Step16 downstream utilization summaries"),
]


def step_names() -> list[str]:
    return [step.name for step in STEP_DEFINITIONS]


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
            "recommended only after lowering batch_size still fails"
            if oom
            else "not required for Step 16 local downstream utilization ablation"
        ),
    }


def is_step16_owned_output_path(path_value: str | Path) -> bool:
    value = str(path_value)
    if any(marker in value for marker in PROTECTED_MARKERS):
        return False
    return "downstream_utilization_bair_1000_128" in value


def validate_step16_output_paths(config: dict[str, Any]) -> None:
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    if not is_step16_owned_output_path(run_dir):
        raise ValueError(f"Refusing non-Step16 run dir: {run_dir}")


def required_input_paths(config: dict[str, Any]) -> list[Path]:
    paths = [
        Path(config["data"]["train_token_shard_dir"]),
        Path(config["data"]["test_token_shard_dir"]),
        Path(config["data"]["train_importance_shard_dir"]),
        Path(config["data"]["test_importance_shard_dir"]),
        Path(config["teacher_reference"]["checkpoint"]),
        Path(config["step15_reference"]["baseline_aggregate"]),
        Path(config["step15_reference"]["validation_summary"]),
    ]
    paths.extend(selector_checkpoint_path(config, int(seed)) for seed in config["learned_selectors"]["seeds"])
    return paths


def validate_required_inputs(config: dict[str, Any]) -> None:
    missing = [str(path) for path in required_input_paths(config) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fixed Step15 inputs: {missing}")


def select_phase_a_top_variants(
    rows: list[dict[str, Any]],
    top_n: int = 2,
    metric: str = "student_future_mse",
    lower_is_better: bool = True,
) -> list[str]:
    successful = [row for row in rows if bool(row.get("success", True)) and _is_finite(row.get(metric))]
    successful = sorted(successful, key=lambda row: float(row[metric]), reverse=not lower_is_better)
    return [str(row["variant"]) for row in successful[: int(top_n)]]


def write_partial_summary(
    run_dir: str | Path,
    *,
    current_step: str,
    completed_steps: list[str],
    status: str,
    error: str | None = None,
    resource: dict[str, Any] | None = None,
    selected_phase_b_variants: list[str] | None = None,
) -> Path:
    path = Path(run_dir) / "partial_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "downstream_utilization_bair_1000_128",
        "status": status,
        "current_step": current_step,
        "completed_steps": completed_steps,
        "error": error,
        "resource": resource or {},
        "selected_phase_b_variants": selected_phase_b_variants or [],
        "updated_at_unix": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _variant_by_name(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(variant["name"]): dict(variant) for variant in config["phase_a"]["variants"]}


def _phase_a_run_dir(run_dir: Path, variant_name: str, seed: int) -> Path:
    return run_dir / "phase_a" / f"{variant_name}_seed{int(seed)}"


def _phase_b_run_dir(run_dir: Path, variant_name: str, seed: int) -> Path:
    return run_dir / "phase_b" / f"{variant_name}_seed{int(seed)}"


def _run_one_variant(
    config: dict[str, Any],
    *,
    phase: str,
    variant: dict[str, Any],
    seed: int,
    max_steps: int,
    batch_size: int,
    num_workers: int,
    run_dir: Path,
) -> dict[str, Any]:
    try:
        return train_downstream_utilization_variant(
            config,
            phase=phase,
            variant=variant,
            seed=seed,
            max_steps=max_steps,
            batch_size=batch_size,
            num_workers=num_workers,
            run_dir=run_dir,
        )
    except BaseException as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return write_failed_variant_summary(
            run_dir=run_dir,
            phase=phase,
            variant=variant,
            seed=seed,
            error=repr(exc),
        )


def run_downstream_utilization_ablation(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_yaml(config_path)
    validate_step16_output_paths(config)
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    before = resource_snapshot()
    ensure_resource_limits(config, before)
    validate_required_inputs(config)

    start = time.perf_counter()
    completed: list[str] = []
    selected_phase_b: list[str] = []
    oom = False
    current_step = "resource_check"
    try:
        completed.append("resource_check")
        write_partial_summary(run_dir, current_step=current_step, completed_steps=completed, status="running", resource={"before": before})

        current_step = "phase_a"
        phase_a_cfg = config["phase_a"]
        phase_a_rows: list[dict[str, Any]] = []
        for seed in [int(value) for value in phase_a_cfg.get("seeds", [0])]:
            for variant in phase_a_cfg["variants"]:
                print(f"RUN_STEP16_PHASE_A variant={variant['name']} seed={seed}", flush=True)
                row = _run_one_variant(
                    config,
                    phase="phase_a",
                    variant=variant,
                    seed=seed,
                    max_steps=int(phase_a_cfg["max_steps"]),
                    batch_size=int(phase_a_cfg["batch_size"]),
                    num_workers=int(phase_a_cfg.get("num_workers", 0)),
                    run_dir=_phase_a_run_dir(run_dir, str(variant["name"]), seed),
                )
                phase_a_rows.append(row)
                if "out of memory" in str(row.get("error", "")).lower() or "oom" in str(row.get("error", "")).lower():
                    oom = True
                after_step = resource_snapshot()
                ensure_resource_limits(config, after_step)
        completed.append("phase_a")
        write_partial_summary(run_dir, current_step=current_step, completed_steps=completed, status="running", resource={"before": before, "latest": resource_snapshot()})

        current_step = "select_phase_b"
        phase_b_cfg = config["phase_b"]
        selected_phase_b = select_phase_a_top_variants(
            phase_a_rows,
            top_n=int(phase_b_cfg.get("select_top_n_from_phase_a", 2)),
            metric=str(phase_b_cfg.get("selection_metric", "student_future_mse")),
            lower_is_better=bool(phase_b_cfg.get("lower_is_better", True)),
        )
        if len(selected_phase_b) < int(phase_b_cfg.get("select_top_n_from_phase_a", 2)):
            raise RuntimeError(f"Not enough successful Phase A variants for Phase B: {selected_phase_b}")
        completed.append("select_phase_b")
        write_partial_summary(
            run_dir,
            current_step=current_step,
            completed_steps=completed,
            status="running",
            resource={"before": before, "latest": resource_snapshot()},
            selected_phase_b_variants=selected_phase_b,
        )

        current_step = "phase_b"
        if bool(phase_b_cfg.get("enabled", True)):
            variant_lookup = _variant_by_name(config)
            for variant_name in selected_phase_b:
                variant = variant_lookup[variant_name]
                for seed in [int(value) for value in phase_b_cfg.get("seeds", [0, 1, 2])]:
                    print(f"RUN_STEP16_PHASE_B variant={variant_name} seed={seed}", flush=True)
                    row = _run_one_variant(
                        config,
                        phase="phase_b",
                        variant=variant,
                        seed=seed,
                        max_steps=int(phase_b_cfg["max_steps"]),
                        batch_size=int(phase_b_cfg["batch_size"]),
                        num_workers=int(phase_b_cfg.get("num_workers", 0)),
                        run_dir=_phase_b_run_dir(run_dir, variant_name, seed),
                    )
                    if "out of memory" in str(row.get("error", "")).lower() or "oom" in str(row.get("error", "")).lower():
                        oom = True
                    after_step = resource_snapshot()
                    ensure_resource_limits(config, after_step)
        completed.append("phase_b")

        current_step = "write_summary"
        after = resource_snapshot()
        resource = _resource_delta(before, after, time.perf_counter() - start, oom=oom)
        summary = write_downstream_utilization_summaries(
            run_dir=run_dir,
            config=config,
            step15_aggregate_path=config["step15_reference"]["baseline_aggregate"],
            resource_summary=resource,
        )
        completed.append("write_summary")
        write_partial_summary(
            run_dir,
            current_step="complete",
            completed_steps=completed,
            status="complete",
            resource=resource,
            selected_phase_b_variants=selected_phase_b,
        )
        print("BAIR_DOWNSTREAM_UTILIZATION_ABLATION_RUNNER_COMPLETE = true", flush=True)
        return summary
    except BaseException as exc:
        oom = oom or isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower() or "oom" in str(exc).lower()
        if torch.cuda.is_available():
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
            selected_phase_b_variants=selected_phase_b,
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_downstream_utilization_ablation(args.config)
    print("BAIR_DOWNSTREAM_UTILIZATION_ABLATION_RUNNER_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
