"""Step28 read-only context sanity analysis for BridgeData V2 TFDS artifacts."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml

from analysis.bridgedata_v2_tfds_teacher_planning import (
    PRIMARY_STEP29_RECOMMENDATION,
    build_next_step_decision,
    build_teacher_plan,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "bridgedata_v2_tfds_context_sanity_step28.yaml"
PROXY_LABEL_QUALITY_NOTE = "proxy dry-run only; not final teacher label"


def run_context_sanity_analysis(
    config_path: str | Path = DEFAULT_CONFIG,
    write_outputs: bool = True,
    artifact_payloads: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the Step28 analysis without download, extraction, generation, or training."""

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(config_path)
    env_guard = _env_guard()
    paths = _output_paths(config)
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        payload = _safe_stop_payload(config, "env_isaaclab is polluted with TensorFlow/TFDS", env_guard)
        if write_outputs:
            _write_outputs(paths, payload)
        return payload

    missing = [] if artifact_payloads is not None else missing_required_inputs(config)
    if missing:
        payload = _safe_stop_payload(config, f"missing Step24-27 artifacts: {missing}", env_guard)
        payload["summary"]["missing_inputs"] = missing
        if write_outputs:
            _write_outputs(paths, payload)
        return payload

    artifacts = artifact_payloads or load_artifact_payloads(config)
    payload = build_context_sanity_diagnostics(config, artifacts, env_guard=env_guard)
    if write_outputs:
        _write_outputs(paths, payload)
    return payload


def load_artifact_payloads(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    input_cfg = config["input"]
    return {
        "step24_summary": _read_json(input_cfg["step24_token_summary_json"]),
        "step25_summary": _read_json(input_cfg["step25_importance_summary_json"]),
        "step26_summary": _read_json(input_cfg["step26_smoke_summary_json"]),
        "step26_policy_metrics": _read_json(input_cfg["step26_policy_metrics_json"]),
        "step27_summary": _read_json(input_cfg["step27_training_summary_json"]),
        "step27_loss_curves": _read_json(input_cfg["step27_loss_curves_json"]),
        "step27_policy_comparison": _read_json(input_cfg["step27_policy_comparison_json"]),
        "step27_final_eval": _read_json(input_cfg["step27_final_eval_json"]),
    }


def missing_required_inputs(config: dict[str, Any]) -> list[str]:
    input_cfg = config["input"]
    required = [
        input_cfg["step24_token_summary_json"],
        input_cfg["step24_token_manifest_jsonl"],
        input_cfg["step25_importance_summary_json"],
        input_cfg["step25_importance_manifest_jsonl"],
        input_cfg["step26_smoke_summary_json"],
        input_cfg["step26_policy_metrics_json"],
        input_cfg["step27_training_summary_json"],
        input_cfg["step27_loss_curves_json"],
        input_cfg["step27_policy_comparison_json"],
        input_cfg["step27_final_eval_json"],
    ]
    return [str(path) for path in required if not Path(path).exists()]


def build_context_sanity_diagnostics(
    config: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    env_guard: dict[str, bool] | None = None,
) -> dict[str, Any]:
    env_guard = env_guard or {"tensorflow": False, "tensorflow_datasets": False}
    thresholds = config.get("diagnostic_thresholds", {})
    step24 = artifacts.get("step24_summary", {})
    step25 = artifacts.get("step25_summary", {})
    step26 = artifacts.get("step26_summary", {})
    step26_policies = artifacts.get("step26_policy_metrics", {})
    step27 = artifacts.get("step27_summary", {})
    step27_comparison = artifacts.get("step27_policy_comparison", {})

    sample_count = _sample_count(step25, step26, step27)
    pipeline = _pipeline_completeness(step24, step25, step26, step27)
    current_only = _current_only_overfit_diagnosis(step27_comparison, thresholds)
    context_utility = _context_utility_diagnosis(
        step26,
        step26_policies,
        step27,
        step27_comparison,
        sample_count,
        current_only,
        thresholds,
    )
    importance_label = _importance_label_diagnosis(step25, context_utility)
    memorization = _memorization_risk_diagnosis(sample_count, current_only, thresholds)
    current_importance = _current_importance_decision()
    teacher_plan = build_teacher_plan(sample_count)
    next_step = build_next_step_decision(sample_count)

    step24_pass = pipeline["step24_token_extraction_performed"]
    step25_pass = pipeline["step25_importance_generation_performed"]
    step26_pass = pipeline["step26_world_model_smoke_performed"]
    step27_pass = pipeline["step27_tiny_overfit_training_performed"]
    all_previous_steps_pass = step24_pass and step25_pass and step26_pass and step27_pass
    safe_stop = not all_previous_steps_pass
    summary = {
        "stage": config["stage"],
        "context_sanity_performed": not safe_stop,
        "safe_stop": safe_stop,
        "reason": None if not safe_stop else "Step24-27 summaries did not all report pass-like completion.",
        "step24_pass": step24_pass,
        "step25_pass": step25_pass,
        "step26_pass": step26_pass,
        "step27_pass": step27_pass,
        "pipeline_completeness": pipeline,
        "num_samples": sample_count,
        "current_only_can_overfit": current_only["current_only_can_overfit"],
        "current_only_relative_loss_decrease": current_only["current_only_relative_loss_decrease"],
        "memorization_risk": memorization["memorization_risk"],
        "proxy_importance_mass_advantage_over_random": context_utility[
            "proxy_importance_mass_advantage_over_random"
        ],
        "context_utility_claim_allowed": context_utility["context_utility_claim_allowed"],
        "importance_label_is_proxy_only": importance_label["label_is_proxy"],
        "train_current_importance_now": current_importance["train_current_importance_now"],
        "allow_current_importance_diagnostic_later": current_importance[
            "allow_current_importance_diagnostic_later"
        ],
        "recommended_step29": PRIMARY_STEP29_RECOMMENDATION,
        "alternative_step29": next_step["alternative_step29"],
        "download_performed": False,
        "model_download_performed": False,
        "training_performed": False,
        "optimizer_step_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "videomae_training_performed": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "current_importance_training_performed": False,
        "world_model_training_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "checkpoint_saved": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
        "vlm_connected": False,
        "rl_connected": False,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": _safety_gate_pass(env_guard),
    }
    return {
        "summary": summary,
        "context_utility": context_utility,
        "memorization_risk": memorization,
        "importance_label": importance_label,
        "teacher_plan": teacher_plan,
        "next_step_decision": next_step,
        "current_only_overfit": current_only,
        "current_importance_decision": current_importance,
    }


def _pipeline_completeness(
    step24: dict[str, Any],
    step25: dict[str, Any],
    step26: dict[str, Any],
    step27: dict[str, Any],
) -> dict[str, bool]:
    return {
        "step24_token_extraction_performed": bool(step24.get("token_extraction_performed"))
        and not bool(step24.get("safe_stop")),
        "step25_importance_generation_performed": bool(step25.get("importance_generation_performed"))
        and not bool(step25.get("safe_stop")),
        "step26_world_model_smoke_performed": bool(step26.get("world_model_smoke_performed"))
        and not bool(step26.get("safe_stop")),
        "step27_tiny_overfit_training_performed": bool(step27.get("tiny_overfit_training_performed"))
        and not bool(step27.get("safe_stop")),
    }


def _current_only_overfit_diagnosis(
    step27_comparison: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    current = _policy_metric(step27_comparison, "current_only")
    relative_decrease = _float(current.get("relative_loss_decrease"))
    high_threshold = float(thresholds.get("current_only_relative_decrease_high", 0.95))
    can_overfit = bool(relative_decrease >= high_threshold)
    return {
        "current_only_can_overfit": can_overfit,
        "current_only_relative_loss_decrease": relative_decrease,
        "initial_loss": _float(current.get("initial_loss")),
        "final_loss": _float(current.get("final_loss")),
        "best_loss": _float(current.get("best_loss")),
        "threshold": high_threshold,
        "interpretation": (
            "With only 4 samples, current-only can memorize/predict the short-horizon future summary; "
            "this prevents claiming context advantage from Step27."
        ),
    }


def _context_utility_diagnosis(
    step26_summary: dict[str, Any],
    step26_policy_payload: dict[str, Any],
    step27_summary: dict[str, Any],
    step27_comparison: dict[str, Any],
    sample_count: int,
    current_only: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    step26_metrics = step26_policy_payload or {"policy_metrics": step26_summary.get("policy_metric_table", [])}
    random26 = _policy_metric(step26_metrics, "random_context_topk")
    proxy26 = _policy_metric(step26_metrics, "proxy_importance_topk")
    full26 = _policy_metric(step26_metrics, "full_context_reference")
    current27 = _policy_metric(step27_comparison, "current_only")
    random27 = _policy_metric(step27_comparison, "random_context_topk")
    proxy27 = _policy_metric(step27_comparison, "proxy_importance_topk")
    full27 = _policy_metric(step27_comparison, "full_context_reference")

    random_mass = _float(random26.get("mean_selected_importance_mass"))
    proxy_mass = _float(proxy26.get("mean_selected_importance_mass"))
    full_mass = _float(full26.get("mean_selected_importance_mass"))
    mass_advantage = proxy_mass / random_mass if random_mass > 0.0 else None
    concentration_threshold = float(thresholds.get("proxy_importance_mass_advantage_min", 3.0))
    proxy_final_loss_better_than_random = _float(proxy27.get("final_loss")) < _float(random27.get("final_loss"))
    full_best = _float(full27.get("best_loss"))
    proxy_best = _float(proxy27.get("best_loss"))
    close_gap = abs(proxy_best - full_best) / max(abs(full_best), 1.0e-12)
    close_threshold = float(thresholds.get("proxy_best_loss_close_to_full_relative_gap", 0.25))
    proxy_best_close = bool(math.isfinite(close_gap) and close_gap <= close_threshold)
    claim_allowed = False
    return {
        "sample_count": int(sample_count),
        "step26_selected_importance_mass": {
            "current_only": _float(_policy_metric(step26_metrics, "current_only").get("mean_selected_importance_mass")),
            "random_context_topk": random_mass,
            "proxy_importance_topk": proxy_mass,
            "full_context_reference": full_mass,
        },
        "step27_losses": {
            "current_only": _loss_fields(current27),
            "random_context_topk": _loss_fields(random27),
            "proxy_importance_topk": _loss_fields(proxy27),
            "full_context_reference": _loss_fields(full27),
        },
        "proxy_importance_mass_advantage_over_random": _nullable_float(mass_advantage),
        "proxy_topk_selects_concentrated_context": bool(
            mass_advantage is not None and mass_advantage >= concentration_threshold
        ),
        "proxy_final_loss_better_than_random": bool(proxy_final_loss_better_than_random),
        "proxy_best_loss_close_to_full": proxy_best_close,
        "proxy_best_loss_relative_gap_to_full": float(close_gap) if math.isfinite(close_gap) else None,
        "current_only_can_overfit": bool(current_only["current_only_can_overfit"]),
        "context_utility_claim_allowed": claim_allowed,
        "reason": (
            "selected importance mass is concentrated, but tiny-overfit loss does not prove predictive "
            "advantage because sample count is 4 and model is random-init/tiny."
        ),
        "allowed_claim": "proxy importance has ranking concentration, not validated predictive utility yet",
        "step27_loss_quality_note": step27_summary.get("loss_quality_note"),
    }


def _importance_label_diagnosis(step25: dict[str, Any], context_utility: dict[str, Any]) -> dict[str, Any]:
    raw_mean = abs(_float(step25.get("importance_raw_mean")))
    raw_std = abs(_float(step25.get("importance_raw_std")))
    norm_min = _float(step25.get("importance_norm_min"))
    norm_max = _float(step25.get("importance_norm_max"))
    method = str(step25.get("method") or "")
    label_note = str(step25.get("label_quality_note") or PROXY_LABEL_QUALITY_NOTE)
    return {
        "importance_method": method,
        "label_is_proxy": method == "proxy_token_mse_dryrun" or "proxy" in label_note,
        "label_is_final_teacher": False,
        "raw_signal_small": bool(max(raw_mean, raw_std) < 1.0e-4),
        "importance_raw_mean": raw_mean,
        "importance_raw_std": raw_std,
        "importance_norm_min": norm_min,
        "importance_norm_max": norm_max,
        "normalized_signal_present": bool(norm_min <= 0.0 and norm_max >= 1.0),
        "selected_mass_advantage_present": bool(
            (context_utility.get("proxy_importance_mass_advantage_over_random") or 0.0) > 1.0
        ),
        "label_quality_note": label_note,
        "recommended_action": "use only for smoke; design stronger teacher before scaling selector training",
    }


def _memorization_risk_diagnosis(
    sample_count: int,
    current_only: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    tiny_threshold = int(thresholds.get("tiny_sample_count_threshold", 8))
    high = sample_count <= tiny_threshold and bool(current_only["current_only_can_overfit"])
    return {
        "sample_count": int(sample_count),
        "memorization_risk": "high" if high else "medium",
        "reason": "All policies, including current_only, can overfit almost perfectly on 4 samples."
        if high
        else "Sample size or current-only overfit signal is not enough to mark high risk.",
        "generalization_not_tested": True,
        "requires_train_val_split": True,
    }


def _current_importance_decision() -> dict[str, Any]:
    return {
        "train_current_importance_now": False,
        "allow_current_importance_diagnostic_later": True,
        "reason": (
            "Current tokens are the present state and should remain full until context bottleneck utility is validated."
        ),
    }


def _safe_stop_payload(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    sample_count = 0
    teacher_plan = build_teacher_plan(sample_count)
    next_step = build_next_step_decision(sample_count)
    summary = {
        "stage": config["stage"],
        "context_sanity_performed": False,
        "safe_stop": True,
        "reason": reason,
        "step24_pass": False,
        "step25_pass": False,
        "step26_pass": False,
        "step27_pass": False,
        "num_samples": sample_count,
        "current_only_can_overfit": False,
        "current_only_relative_loss_decrease": 0.0,
        "memorization_risk": "unknown",
        "proxy_importance_mass_advantage_over_random": None,
        "context_utility_claim_allowed": False,
        "importance_label_is_proxy_only": False,
        "train_current_importance_now": False,
        "allow_current_importance_diagnostic_later": True,
        "recommended_step29": PRIMARY_STEP29_RECOMMENDATION,
        "alternative_step29": next_step["alternative_step29"],
        "download_performed": False,
        "model_download_performed": False,
        "training_performed": False,
        "optimizer_step_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "videomae_training_performed": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "current_importance_training_performed": False,
        "world_model_training_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "checkpoint_saved": False,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": _safety_gate_pass(env_guard),
    }
    return {
        "summary": summary,
        "context_utility": {"context_utility_claim_allowed": False, "reason": reason},
        "memorization_risk": {
            "sample_count": sample_count,
            "memorization_risk": "unknown",
            "generalization_not_tested": True,
            "requires_train_val_split": True,
        },
        "importance_label": {
            "importance_method": None,
            "label_is_proxy": False,
            "label_is_final_teacher": False,
            "recommended_action": "safe-stop until Step24-27 summaries are present",
        },
        "teacher_plan": teacher_plan,
        "next_step_decision": next_step,
        "current_only_overfit": {"current_only_can_overfit": False, "current_only_relative_loss_decrease": 0.0},
        "current_importance_decision": _current_importance_decision(),
    }


def _write_outputs(paths: dict[str, Path], payload: dict[str, Any]) -> None:
    _write_json(paths["sanity_summary_json"], payload["summary"])
    _write_json(paths["context_utility_json"], payload["context_utility"])
    _write_json(paths["memorization_risk_json"], payload["memorization_risk"])
    _write_json(paths["importance_label_json"], payload["importance_label"])
    _write_json(paths["teacher_plan_json"], payload["teacher_plan"])
    _write_json(paths["next_step_decision_json"], payload["next_step_decision"])


def _output_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {key: Path(value) for key, value in config["output"].items() if key.endswith("_json")}


def _policy_metric(payload: dict[str, Any], policy: str) -> dict[str, Any]:
    metrics = payload.get("policy_metrics") or payload.get("policy_metric_table") or []
    for item in metrics:
        if str(item.get("policy")) == policy:
            return dict(item)
    return {}


def _loss_fields(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "initial_loss": _float(metric.get("initial_loss")),
        "final_loss": _float(metric.get("final_loss")),
        "best_loss": _float(metric.get("best_loss")),
        "relative_loss_decrease": _float(metric.get("relative_loss_decrease")),
    }


def _sample_count(*summaries: dict[str, Any]) -> int:
    for summary in reversed(summaries):
        value = summary.get("num_samples")
        if value is not None:
            return int(value)
    return 0


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _nullable_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _safety_gate_pass(env_guard: dict[str, bool]) -> bool:
    return not bool(env_guard.get("tensorflow")) and not bool(env_guard.get("tensorflow_datasets"))


def _env_guard() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data

