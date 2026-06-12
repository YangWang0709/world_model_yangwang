"""Step34 planning logic for proxy-supervised temporal selector scaffolding."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import yaml

from data.bridgedata_v2_tfds_world_model_batch import load_bridge_tfds_world_model_samples
from models.bridgedata_v2_proxy_temporal_selector import (
    ProxyTemporalSelectorHead,
    proxy_temporal_selector_dry_run,
)


STAGE = "bridgedata_v2_tfds_proxy_temporal_selector_step34"
REQUIRED_STEP33B_KEYS = (
    "step33b_eval_json",
    "step33b_decision_json",
    "step33b_dataset_bias_json",
    "step33b_within_shard_results_json",
    "step33b_cross_shard_results_json",
    "step33b_mixed_shard_results_json",
    "step33b_shard_split_json",
    "step33b_token_manifest_jsonl",
    "step33b_token_summary_json",
    "step33b_importance_manifest_jsonl",
    "step33b_importance_summary_json",
)


def prepare_step34_proxy_temporal_selector(
    config_path: str | Path,
    *,
    run_dry_run: bool | None = None,
) -> dict[str, Any]:
    config = load_step34_config(config_path)
    payload = build_step34_selector_plan(config, run_dry_run=run_dry_run)
    write_step34_outputs(config, payload)
    return payload


def build_step34_selector_plan(
    config: dict[str, Any],
    *,
    run_dry_run: bool | None = None,
) -> dict[str, Any]:
    env_guard = env_isaaclab_tensorflow_guard()
    missing = missing_step33b_inputs(config)
    if missing:
        return _safe_stop(config, env_guard, f"missing Step33B inputs: {missing}")

    step33b = _load_step33b_bundle(config)
    evidence = summarize_step33b_evidence(step33b)
    split_plan = build_step34_split_plan(step33b["splits"], step33b)
    leakage = summarize_leakage_checks(step33b["splits"], step33b["dataset_bias"])
    gates = build_future_training_gates(config, evidence, leakage, step33b["dataset_bias"])
    dry_run_enabled = bool(config.get("dry_run", {}).get("enabled", True)) if run_dry_run is None else bool(run_dry_run)
    dry_run = run_selector_forward_dry_run(config) if dry_run_enabled else _skipped_dry_run()
    safety_gate = step34_safety_gate(config, env_guard, evidence, leakage, dry_run)
    decision = {
        "stage": STAGE,
        "may_prepare_proxy_supervised_selector_training": True,
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "final_selector_training_allowed": False,
        "downstream_task_improvement_evidence_allowed": False,
        "future_selector_training_gate_ready": False,
        "reason_selector_training_not_allowed": "Step34 is planning/scaffold only; future gates are defined but not satisfied by a trained selector.",
        "recommended_step35": {
            "name": "review Step34 gates before any bounded proxy-supervised selector training",
            "condition": "explicit approval plus leakage/bias gates plus dry-run scaffold pass",
            "scope": "no final downstream selector and no current-importance training by default",
        },
        "safety_gate_pass": bool(safety_gate),
    }
    return {
        "stage": STAGE,
        "safe_stop": False,
        "reason": None,
        "env_isaaclab_guard": env_guard,
        "step33b_evidence": evidence,
        "split_plan": split_plan,
        "leakage_checks": leakage,
        "future_training_gates": gates,
        "selector_scaffold": selector_scaffold_summary(config),
        "dry_run_summary": dry_run,
        "decision": decision,
        "safety_gate_pass": bool(safety_gate),
    }


def summarize_step33b_evidence(step33b: dict[str, Any]) -> dict[str, Any]:
    decision = step33b["decision"]
    dataset_bias = step33b["dataset_bias"]
    within = step33b["within"]
    cross = step33b["cross"]
    mixed = step33b["mixed"]
    importance = step33b["importance_summary"]
    return {
        "proxy_signal_stable_across_shards": bool(decision.get("proxy_signal_stable_across_shards", False)),
        "shard1_within_proxy_beats_current_fraction": float(within.get("proxy_beats_current_fraction", 0.0)),
        "shard1_within_proxy_beats_random_fraction": float(within.get("proxy_beats_random_fraction", 0.0)),
        "shard1_within_proxy_gain_over_current_mean": float(within.get("proxy_gain_over_current_mean", 0.0)),
        "cross_proxy_beats_current_fraction": float(cross.get("proxy_beats_current_fraction", 0.0)),
        "cross_proxy_beats_random_fraction": float(cross.get("proxy_beats_random_fraction", 0.0)),
        "cross_proxy_gain_over_current_mean": float(cross.get("proxy_gain_over_current_mean", 0.0)),
        "mixed_proxy_beats_current_fraction": float(mixed.get("proxy_beats_current_fraction", 0.0)),
        "mixed_proxy_beats_random_fraction": float(mixed.get("proxy_beats_random_fraction", 0.0)),
        "mixed_proxy_gain_over_current_mean": float(mixed.get("proxy_gain_over_current_mean", 0.0)),
        "full_context_noise_confirmed": bool(
            within.get("full_context_noise_confirmed")
            or cross.get("full_context_noise_confirmed")
            or mixed.get("full_context_noise_confirmed")
        ),
        "dataset_bias_detected": bool(dataset_bias.get("dataset_bias_detected", False)),
        "distribution_shift_flags": dict(dataset_bias.get("distribution_shift_flags") or {}),
        "language_hash_comparable": bool(dataset_bias.get("language_hash_comparable", False)),
        "language_hash_non_comparable_not_shift": not bool(
            (dataset_bias.get("distribution_shift_flags") or {}).get("language_hash_shift", False)
        ),
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_token_shape": [16, 392, 768],
        "current_token_shape": [4, 392, 768],
        "future_token_shape": [4, 392, 768],
        "proxy_importance_shape": [16, 392],
        "topk_mass_mean": float(importance.get("topk_mass_mean", 0.0)),
    }


def build_step34_split_plan(splits: dict[str, Any], step33b: dict[str, Any]) -> dict[str, Any]:
    split_rows = splits.get("splits", [])
    seeds = [int(seed) for seed in splits.get("split_seeds", [])]
    return {
        "strict_shard_aware_splits": True,
        "split_seeds": seeds,
        "within_shard_validation": {
            "enabled": True,
            "train_shard": "shard1",
            "val_shard": "shard1",
            "num_seed_splits": len(split_rows),
            "trajectory_disjoint_all_splits": all(bool(split.get("train_val_trajectory_disjoint")) for split in split_rows),
            "sample_id_disjoint_all_splits": all(_ids_disjoint(split) for split in split_rows),
        },
        "cross_shard_validation": {
            "enabled": True,
            "planned_directions": ["train_shard0_val_shard1", "train_shard1_val_shard0"],
            "step33b_eval_performed": int(step33b["cross"].get("num_rows", 0)) > 0,
            "requires_no_shared_sample_ids_across_shards": True,
        },
        "mixed_shard_validation": {
            "enabled": True,
            "planned_setting": "mixed_train_mixed_val_shard_aware",
            "step33b_eval_performed": int(step33b["mixed"].get("num_rows", 0)) > 0,
            "requires_per_shard_breakdown": True,
        },
        "no_downstream_task_improvement_used_as_evidence": True,
    }


def summarize_leakage_checks(splits: dict[str, Any], dataset_bias: dict[str, Any]) -> dict[str, Any]:
    split_rows = splits.get("splits", [])
    sample_overlap = [split.get("split_seed") for split in split_rows if not _ids_disjoint(split)]
    trajectory_overlap = [
        split.get("split_seed")
        for split in split_rows
        if set(split.get("train_trajectories", [])) & set(split.get("val_trajectories", []))
    ]
    shift_flags = dict(dataset_bias.get("distribution_shift_flags") or {})
    return {
        "train_val_sample_id_overlap_detected": bool(sample_overlap),
        "sample_overlap_split_seeds": sample_overlap,
        "train_val_trajectory_overlap_detected": bool(trajectory_overlap),
        "trajectory_overlap_split_seeds": trajectory_overlap,
        "language_hash_comparable": bool(dataset_bias.get("language_hash_comparable", False)),
        "language_hash_shift_detected": bool(shift_flags.get("language_hash_shift", False)),
        "language_hash_non_comparable_not_treated_as_shift": not bool(shift_flags.get("language_hash_shift", False)),
        "dataset_bias_detected": bool(dataset_bias.get("dataset_bias_detected", False)),
        "distribution_shift_flags": shift_flags,
        "raw_language_text_saved": bool(dataset_bias.get("raw_language_text_saved", False)),
        "no_language_or_trajectory_leakage": not sample_overlap
        and not trajectory_overlap
        and not bool(shift_flags.get("language_hash_shift", False)),
        "safety_gate_pass": not sample_overlap and not trajectory_overlap and not bool(dataset_bias.get("dataset_bias_detected", False)),
    }


def build_future_training_gates(
    config: dict[str, Any],
    evidence: dict[str, Any],
    leakage: dict[str, Any],
    dataset_bias: dict[str, Any],
) -> dict[str, Any]:
    del config
    no_shift = not bool(dataset_bias.get("dataset_bias_detected", False)) and not any(
        bool(value) for value in (dataset_bias.get("distribution_shift_flags") or {}).values()
    )
    return {
        "selector_beats_random_baseline": {
            "required_for_future_training": True,
            "status": "pending",
            "evidence_source": "future proxy-supervised selector validation, not Step34 dry-run",
        },
        "selector_beats_current_only_baseline": {
            "required_for_future_training": True,
            "status": "pending",
            "evidence_source": "future proxy-supervised selector validation, not downstream task improvement",
        },
        "selector_generalizes_across_shard": {
            "required_for_future_training": True,
            "status": "pending",
            "planned_evals": ["within_shard", "cross_shard", "mixed_shard"],
        },
        "no_language_or_trajectory_leakage": {
            "required_for_future_training": True,
            "status": "pass" if leakage["no_language_or_trajectory_leakage"] else "fail",
        },
        "no_dataset_bias_or_shard_shift": {
            "required_for_future_training": True,
            "status": "pass" if no_shift else "fail",
        },
        "full_context_noisy_issue_acknowledged": {
            "required_for_future_training": True,
            "status": "pass" if evidence["full_context_noise_confirmed"] else "fail",
        },
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
    }


def run_selector_forward_dry_run(config: dict[str, Any]) -> dict[str, Any]:
    loader_config = {
        "input": {
            "token_manifest_jsonl": config["input"]["step33b_token_manifest_jsonl"],
            "token_summary_json": config["input"]["step33b_token_summary_json"],
            "token_smoke_dir": str(Path(config["input"]["step33b_token_manifest_jsonl"]).parent / "token_smoke"),
            "importance_manifest_jsonl": config["input"]["step33b_importance_manifest_jsonl"],
            "importance_summary_json": config["input"]["step33b_importance_summary_json"],
            "importance_smoke_dir": str(Path(config["input"]["step33b_importance_manifest_jsonl"]).parent / "importance_smoke"),
        },
        "sample_limits": {"max_samples": int(config["dry_run"].get("max_samples", 1))},
    }
    samples = load_bridge_tfds_world_model_samples(loader_config)
    selector_cfg = config["selector_scaffold"]
    selector = ProxyTemporalSelectorHead(
        token_dim=int(selector_cfg.get("token_dim", 768)),
        hidden_dim=int(selector_cfg.get("hidden_dim", 128)),
        context_frames=int(selector_cfg.get("context_frames", 16)),
        spatial_tokens=int(selector_cfg.get("spatial_tokens", 392)),
        condition_on_current_summary=bool(selector_cfg.get("condition_on_current_summary", True)),
        dropout=float(selector_cfg.get("dropout", 0.0)),
        detach_token_inputs=bool(selector_cfg.get("detach_token_inputs", True)),
    )
    dry_run = proxy_temporal_selector_dry_run(selector=selector, samples=samples)
    dry_run.update(
        {
            "stage": STAGE,
            "smoke_mode": True,
            "used_existing_local_artifacts_only": True,
            "tensorflow_required": False,
            "tensorflow_datasets_required": False,
            "model_download_performed": False,
            "new_dataset_download_performed": False,
            "current_importance_labels_required": False,
            "safety_gate_pass": bool(dry_run["all_losses_finite"]),
        }
    )
    return dry_run


def selector_scaffold_summary(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["selector_scaffold"]
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "module": str(cfg.get("module", "ProxyTemporalSelectorHead")),
        "consumes": str(cfg.get("consumes", "existing_context_temporal_token_features")),
        "predicts": str(cfg.get("predicts", "proxy_temporal_importance_scores")),
        "freeze_videomae": bool(cfg.get("freeze_videomae", True)),
        "detach_token_inputs": bool(cfg.get("detach_token_inputs", True)),
        "train_video_encoder": False,
        "train_final_selector": False,
        "train_current_importance": False,
        "optimizer_step_allowed": False,
        "checkpoint_saved": False,
    }


def step34_safety_gate(
    config: dict[str, Any],
    env_guard: dict[str, bool],
    evidence: dict[str, Any],
    leakage: dict[str, Any],
    dry_run: dict[str, Any],
) -> bool:
    guards = config["guards"]
    decision = config["decision"]
    return (
        not env_guard["tensorflow_required"]
        and not env_guard["tensorflow_datasets_required"]
        and bool(evidence["proxy_signal_stable_across_shards"])
        and not bool(evidence["dataset_bias_detected"])
        and not any(bool(value) for value in evidence["distribution_shift_flags"].values())
        and bool(leakage["safety_gate_pass"])
        and bool(dry_run.get("safety_gate_pass", True))
        and bool(guards["no_final_selector_training"])
        and bool(guards["no_current_importance_training"])
        and bool(guards["no_videomae_training"])
        and bool(guards["use_existing_local_tfds_shards_only"])
        and bool(guards["use_existing_local_videomae_only"])
        and not bool(decision["selector_training_allowed"])
        and not bool(decision["current_importance_training_allowed"])
        and not bool(decision["context_utility_claim_allowed"])
    )


def env_isaaclab_tensorflow_guard() -> dict[str, bool]:
    return {
        "tensorflow_required": False,
        "tensorflow_datasets_required": False,
        "tensorflow_available": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets_available": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def missing_step33b_inputs(config: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in REQUIRED_STEP33B_KEYS:
        path = Path(str(config["input"][key]))
        if not path.exists():
            missing.append(str(path))
    model_path = Path(str(config["input"]["local_videomae_model"]))
    if not model_path.exists():
        missing.append(str(model_path))
    return missing


def write_step34_outputs(config: dict[str, Any], payload: dict[str, Any]) -> None:
    output = config["output"]
    run_dir = Path(output["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output["plan_summary_json"], payload)
    if "dry_run_summary" in payload:
        _write_json(output["dry_run_summary_json"], payload["dry_run_summary"])


def load_step34_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Step34 config must be a mapping: {path}")
    if data.get("stage") != STAGE:
        raise ValueError(f"unexpected Step34 stage: {data.get('stage')!r}")
    return data


def _load_step33b_bundle(config: dict[str, Any]) -> dict[str, Any]:
    input_cfg = config["input"]
    return {
        "eval": _read_json(input_cfg["step33b_eval_json"]),
        "decision": _read_json(input_cfg["step33b_decision_json"]),
        "dataset_bias": _read_json(input_cfg["step33b_dataset_bias_json"]),
        "within": _read_json(input_cfg["step33b_within_shard_results_json"]),
        "cross": _read_json(input_cfg["step33b_cross_shard_results_json"]),
        "mixed": _read_json(input_cfg["step33b_mixed_shard_results_json"]),
        "splits": _read_json(input_cfg["step33b_shard_split_json"]),
        "importance_summary": _read_json(input_cfg["step33b_importance_summary_json"]),
    }


def _safe_stop(config: dict[str, Any], env_guard: dict[str, bool], reason: str) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "safe_stop": True,
        "reason": reason,
        "env_isaaclab_guard": env_guard,
        "selector_scaffold": selector_scaffold_summary(config),
        "dry_run_summary": _skipped_dry_run(),
        "decision": {
            "context_utility_claim_allowed": False,
            "selector_training_allowed": False,
            "current_importance_training_allowed": False,
            "final_selector_training_allowed": False,
            "safety_gate_pass": False,
        },
        "safety_gate_pass": False,
    }


def _skipped_dry_run() -> dict[str, Any]:
    return {
        "selector_forward_performed": False,
        "selector_training_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
        "videomae_frozen": True,
        "current_importance_training_performed": False,
        "safety_gate_pass": True,
    }


def _ids_disjoint(split: dict[str, Any]) -> bool:
    return not (set(split.get("train_sample_ids", [])) & set(split.get("val_sample_ids", [])))


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
