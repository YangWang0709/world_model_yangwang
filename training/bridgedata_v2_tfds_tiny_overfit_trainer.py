"""Step27 tiny overfit trainer for BridgeData V2 TFDS world-model tokens."""

from __future__ import annotations

import importlib.util
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from data.bridgedata_v2_tfds_context_selection import select_context_tokens, summarize_selection
from data.bridgedata_v2_tfds_world_model_batch import (
    load_bridge_tfds_world_model_samples,
    step24_step25_inputs_missing,
    summarize_world_model_samples,
    validate_world_model_sample,
)
from data.bridgedata_v2_tfds_world_model_training_metrics import (
    LOSS_QUALITY_NOTE,
    acceptance_pass,
    build_policy_comparison,
    summarize_loss_curve,
)
from models.bridgedata_v2_context_bottleneck_smoke_model import BridgeDataContextBottleneckSmokePredictor

OPTIMIZER_SCOPE = "tiny_world_model_predictor_only"


def run_tiny_overfit_training(
    config_path: str | Path,
    samples: list[dict[str, Any]] | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(config_path)
    return train_tiny_overfit(config, samples=samples, write_outputs=write_outputs)


def train_tiny_overfit(
    config: dict[str, Any],
    samples: list[dict[str, Any]] | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    env_guard = _env_isaaclab_guard()
    paths = _output_paths(config)
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        payload = _safe_stop_payload(config, "env_isaaclab is polluted with TensorFlow/TFDS", env_guard)
        if write_outputs:
            _write_training_outputs(paths, payload)
        return payload

    if samples is None:
        missing = _input_artifacts_missing(config)
        if missing:
            payload = _safe_stop_payload(config, f"missing Step24/25/26 artifacts: {missing}", env_guard)
            if write_outputs:
                _write_training_outputs(paths, payload)
            return payload
        samples = load_bridge_tfds_world_model_samples(config)

    for sample in samples:
        validate_world_model_sample(sample)
        _assert_sample_tensors_frozen(sample)

    seed = int(config.get("seed", 42))
    _seed_everything(seed)
    tiny_cfg = config["tiny_overfit"]
    acceptance_cfg = config.get("acceptance", {})
    min_relative = float(acceptance_cfg.get("min_relative_loss_decrease", 0.01))
    device = _resolve_device(str(tiny_cfg.get("device", "cpu")), bool(tiny_cfg.get("allow_cpu_fallback", True)))

    policy_results: list[dict[str, Any]] = []
    loss_curves: list[dict[str, Any]] = []
    all_selection_summaries: list[dict[str, Any]] = []
    optimizer_step_performed = False
    optimizer_param_count = 0
    tiny_predictor_param_count = 0

    for policy_index, policy in enumerate(config["policies"]):
        if not bool(policy.get("train", True)):
            continue
        model = BridgeDataContextBottleneckSmokePredictor(
            hidden_dim=int(tiny_cfg.get("hidden_dim", 512)),
            seed=seed + policy_index,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(tiny_cfg.get("learning_rate", 0.001)),
            weight_decay=float(tiny_cfg.get("weight_decay", 0.0)),
        )
        scope = _optimizer_scope(model, optimizer)
        if scope["optimizer_step_scope"] != OPTIMIZER_SCOPE:
            raise RuntimeError(f"optimizer scope must be {OPTIMIZER_SCOPE}: {scope}")
        optimizer_param_count = max(optimizer_param_count, int(scope["optimizer_param_count"]))
        tiny_predictor_param_count = max(tiny_predictor_param_count, int(scope["tiny_predictor_param_count"]))

        dataset = _policy_dataset(samples, policy, device)
        all_selection_summaries.extend(dataset["selection_summaries"])
        curve, performed = _train_policy(
            model=model,
            optimizer=optimizer,
            current_summaries=dataset["current_summaries"],
            context_summaries=dataset["context_summaries"],
            targets=dataset["targets"],
            train_steps=int(tiny_cfg.get("train_steps", 300)),
            eval_every=int(tiny_cfg.get("eval_every", 25)),
            grad_clip_norm=float(tiny_cfg.get("grad_clip_norm", 1.0)),
        )
        optimizer_step_performed = optimizer_step_performed or performed
        loss_metrics = summarize_loss_curve(
            str(policy["name"]),
            curve,
            min_relative_loss_decrease=min_relative,
        )
        policy_result = {
            "policy": str(policy["name"]),
            "topk": dataset["topk"],
            "num_samples": len(samples),
            "num_steps": int(tiny_cfg.get("train_steps", 300)),
            "optimizer_step_performed": bool(performed),
            "optimizer_step_scope": OPTIMIZER_SCOPE,
            "current_tokens_kept_full": True,
            "train_current_importance": False,
            "tiny_overfit_result_not_final_performance": True,
            "mean_selected_importance_mass": dataset["mean_selected_importance_mass"],
            "selection_summaries": dataset["selection_summaries"],
            **loss_metrics,
        }
        policy_results.append(policy_result)
        loss_curves.append({"policy": str(policy["name"]), "topk": dataset["topk"], "curve": curve})

    policy_comparison = build_policy_comparison(policy_results, min_relative_loss_decrease=min_relative)
    batch_summary = summarize_world_model_samples(samples)
    summary = {
        "stage": config["stage"],
        "tiny_overfit_training_performed": bool(samples) and bool(policy_results),
        "training_performed": bool(samples) and bool(policy_results),
        "tiny_overfit_training_only": True,
        "safe_stop": not (bool(samples) and bool(policy_results)),
        "reason": None if samples and policy_results else "No Step27 tiny-overfit records were generated.",
        "num_samples": len(samples),
        "sample_ids": batch_summary.get("sample_ids", []),
        "policies_trained": [str(result["policy"]) for result in policy_results],
        "train_steps": int(tiny_cfg.get("train_steps", 300)),
        "optimizer": str(tiny_cfg.get("optimizer", "adamw")),
        "optimizer_step_performed": bool(optimizer_step_performed),
        "optimizer_step_scope": OPTIMIZER_SCOPE,
        "optimizer_param_count": int(optimizer_param_count),
        "tiny_predictor_param_count": int(tiny_predictor_param_count),
        "all_losses_finite": bool(policy_comparison.get("all_losses_finite")),
        "policies_with_loss_decrease": int(policy_comparison.get("policies_with_loss_decrease") or 0),
        "min_relative_loss_decrease": min_relative,
        "acceptance_pass": False,
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "tiny_world_model_training_performed": bool(samples) and bool(policy_results),
        "current_importance_training_performed": False,
        "videomae_training_performed": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "world_model_large_training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "download_performed": False,
        "model_download_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "checkpoint_saved": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
        "tiny_overfit_result_not_final_performance": True,
        "loss_quality_note": LOSS_QUALITY_NOTE,
        "device": str(device),
        "env_isaaclab_guard": env_guard,
        "policy_metric_table": policy_comparison.get("policy_metrics", []),
        "selection_summaries": all_selection_summaries,
        "safety_gate_pass": True,
    }
    summary["acceptance_pass"] = acceptance_pass(summary, policy_comparison, acceptance_cfg)
    final_eval = {
        "stage": config["stage"],
        "summary": summary,
        "policy_comparison": policy_comparison,
        "loss_quality_note": LOSS_QUALITY_NOTE,
    }
    payload = {
        "summary": summary,
        "loss_curves": {
            "stage": config["stage"],
            "loss_curves": loss_curves,
            "loss_quality_note": LOSS_QUALITY_NOTE,
        },
        "policy_comparison": policy_comparison,
        "final_eval": final_eval,
    }
    if write_outputs:
        _write_training_outputs(paths, payload)
    return payload


def _train_policy(
    model: BridgeDataContextBottleneckSmokePredictor,
    optimizer: torch.optim.Optimizer,
    current_summaries: torch.Tensor,
    context_summaries: torch.Tensor,
    targets: torch.Tensor,
    train_steps: int,
    eval_every: int,
    grad_clip_norm: float,
) -> tuple[list[dict[str, Any]], bool]:
    curve = [{"step": 0, "loss": _evaluate_loss(model, current_summaries, context_summaries, targets)}]
    optimizer_step_performed = False
    for step in range(1, train_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            pred = model.forward_from_summaries(current_summaries, context_summaries)
            loss = F.mse_loss(pred, targets, reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        optimizer_step_performed = True
        if step == 1 or step % max(eval_every, 1) == 0 or step == train_steps:
            curve.append({"step": step, "loss": _evaluate_loss(model, current_summaries, context_summaries, targets)})
    return curve, optimizer_step_performed


def _evaluate_loss(
    model: BridgeDataContextBottleneckSmokePredictor,
    current_summaries: torch.Tensor,
    context_summaries: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        pred = model.forward_from_summaries(current_summaries, context_summaries)
        loss = F.mse_loss(pred, targets, reduction="mean")
    return float(loss.item())


def _policy_dataset(samples: list[dict[str, Any]], policy: dict[str, Any], device: torch.device) -> dict[str, Any]:
    current_summaries = []
    context_summaries = []
    targets = []
    selection_summaries = []
    masses = []
    for sample in samples:
        selection = select_context_tokens(sample, policy)
        current_summary = sample["current_tokens"].detach().to(dtype=torch.float32).mean(dim=(0, 1))
        if selection["selected_context_tokens"].numel() == 0:
            context_summary = torch.zeros_like(current_summary)
        else:
            context_summary = selection["selected_context_tokens"].detach().to(dtype=torch.float32).mean(dim=0)
        target = sample["future_tokens"].detach().to(dtype=torch.float32).mean(dim=(0, 1))
        current_summaries.append(current_summary)
        context_summaries.append(context_summary)
        targets.append(target)
        masses.append(float(selection["selected_importance_mass"]))
        selection_summaries.append({"sample_id": str(sample["sample_id"]), **summarize_selection(selection)})
    return {
        "current_summaries": torch.stack(current_summaries, dim=0).to(device),
        "context_summaries": torch.stack(context_summaries, dim=0).to(device),
        "targets": torch.stack(targets, dim=0).to(device),
        "selection_summaries": selection_summaries,
        "mean_selected_importance_mass": float(sum(masses) / len(masses)) if masses else 0.0,
        "topk": selection_summaries[0]["topk"] if selection_summaries else policy.get("topk"),
    }


def _optimizer_scope(model: BridgeDataContextBottleneckSmokePredictor, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    model_param_ids = {id(param) for param in model.parameters()}
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group.get("params", [])
        if isinstance(param, torch.nn.Parameter)
    }
    external_params = sorted(optimizer_param_ids - model_param_ids)
    return {
        "optimizer_step_scope": OPTIMIZER_SCOPE if not external_params else "external_parameters_present",
        "optimizer_param_count": len(optimizer_param_ids),
        "tiny_predictor_param_count": len(model_param_ids),
        "external_parameter_count": len(external_params),
        "all_optimizer_params_require_grad": all(param.requires_grad for param in model.parameters()),
    }


def _input_artifacts_missing(config: dict[str, Any]) -> list[str]:
    missing = step24_step25_inputs_missing(config)
    step26_summary = Path(config["input"].get("step26_smoke_summary_json", ""))
    if not step26_summary.exists():
        missing.append(str(step26_summary))
    else:
        payload = json.loads(step26_summary.read_text(encoding="utf-8"))
        if bool(payload.get("safe_stop", True)):
            missing.append("Step26 smoke summary reports safe_stop=true")
        if not bool(payload.get("world_model_smoke_performed", False)):
            missing.append("Step26 smoke summary reports world_model_smoke_performed=false")
    return missing


def _safe_stop_payload(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    summary = {
        "stage": config["stage"],
        "tiny_overfit_training_performed": False,
        "training_performed": False,
        "tiny_overfit_training_only": True,
        "safe_stop": True,
        "reason": reason,
        "num_samples": 0,
        "policies_trained": [str(policy["name"]) for policy in config.get("policies", []) if bool(policy.get("train", True))],
        "train_steps": int(config.get("tiny_overfit", {}).get("train_steps", 0)),
        "optimizer": str(config.get("tiny_overfit", {}).get("optimizer", "adamw")),
        "optimizer_step_performed": False,
        "optimizer_step_scope": OPTIMIZER_SCOPE,
        "all_losses_finite": False,
        "policies_with_loss_decrease": 0,
        "min_relative_loss_decrease": float(config.get("acceptance", {}).get("min_relative_loss_decrease", 0.01)),
        "acceptance_pass": False,
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "tiny_world_model_training_performed": False,
        "current_importance_training_performed": False,
        "videomae_training_performed": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "world_model_large_training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "download_performed": False,
        "model_download_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "checkpoint_saved": False,
        "tiny_overfit_result_not_final_performance": True,
        "loss_quality_note": LOSS_QUALITY_NOTE,
        "env_isaaclab_guard": env_guard,
        "policy_metric_table": [],
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }
    policy_comparison = {
        "num_policies": 0,
        "policies_with_loss_decrease": 0,
        "all_losses_finite": False,
        "policy_metrics": [],
        "loss_quality_note": LOSS_QUALITY_NOTE,
    }
    return {
        "summary": summary,
        "loss_curves": {"stage": config["stage"], "loss_curves": [], "loss_quality_note": LOSS_QUALITY_NOTE},
        "policy_comparison": policy_comparison,
        "final_eval": {"stage": config["stage"], "summary": summary, "policy_comparison": policy_comparison},
    }


def _write_training_outputs(paths: dict[str, Path], payload: dict[str, Any]) -> None:
    _write_json(paths["training_summary_json"], payload["summary"])
    _write_json(paths["loss_curves_json"], payload["loss_curves"])
    _write_json(paths["policy_comparison_json"], payload["policy_comparison"])
    _write_json(paths["final_eval_json"], payload["final_eval"])
    paths["training_summary_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["training_summary_md"].write_text(_format_training_summary_md(payload["summary"]), encoding="utf-8")


def _format_training_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# BridgeData V2 TFDS Tiny Overfit Training Summary",
        "",
        f"- tiny_overfit_training_performed: `{str(summary.get('tiny_overfit_training_performed', False)).lower()}`",
        f"- safe_stop: `{str(summary.get('safe_stop', False)).lower()}`",
        f"- acceptance_pass: `{str(summary.get('acceptance_pass', False)).lower()}`",
        f"- num_samples: `{summary.get('num_samples')}`",
        f"- policies_trained: `{summary.get('policies_trained')}`",
        f"- train_steps: `{summary.get('train_steps')}`",
        f"- optimizer_step_performed: `{str(summary.get('optimizer_step_performed', False)).lower()}`",
        f"- optimizer_step_scope: `{summary.get('optimizer_step_scope')}`",
        f"- all_losses_finite: `{str(summary.get('all_losses_finite', False)).lower()}`",
        f"- policies_with_loss_decrease: `{summary.get('policies_with_loss_decrease')}`",
        f"- current_tokens_kept_full: `{str(summary.get('current_tokens_kept_full', False)).lower()}`",
        f"- train_current_importance: `{str(summary.get('train_current_importance', True)).lower()}`",
        f"- checkpoint_saved: `{str(summary.get('checkpoint_saved', True)).lower()}`",
        f"- loss_quality_note: `{summary.get('loss_quality_note')}`",
    ]
    if summary.get("reason"):
        lines.append(f"- reason: `{summary['reason']}`")
    lines.extend(["", "## Policy Metrics", "", "```json", json.dumps(summary.get("policy_metric_table", []), indent=2), "```", ""])
    return "\n".join(lines)


def _output_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {key: Path(value) for key, value in config["output"].items() if key.endswith("_json") or key.endswith("_md")}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _assert_sample_tensors_frozen(sample: dict[str, Any]) -> None:
    for key in ("context_tokens", "current_tokens", "future_tokens", "context_importance"):
        tensor = sample[key]
        if bool(tensor.requires_grad):
            raise ValueError(f"{key} must not require gradients for Step27 tiny overfit")


def _resolve_device(requested: str, allow_cpu_fallback: bool) -> torch.device:
    if requested == "cuda_if_available" and torch.cuda.is_available():
        try:
            torch.empty((1,), device="cuda")
            return torch.device("cuda")
        except Exception:
            if not allow_cpu_fallback:
                raise
    if requested.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.empty((1,), device=requested)
            return torch.device(requested)
        except Exception:
            if not allow_cpu_fallback:
                raise
    return torch.device("cpu")


def _env_isaaclab_guard() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data
