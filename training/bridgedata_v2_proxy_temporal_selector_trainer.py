"""Bounded Step35 proxy-temporal selector smoke trainer."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import torch

from data.bridgedata_v2_proxy_temporal_selector_dataset_step35 import (
    OPTIMIZER_SCOPE,
    STAGE,
    batch_step35_samples,
    load_step35_config,
    load_step35_samples,
    missing_step35_inputs,
    sample_ids_by_shard_from_splits,
)
from data.bridgedata_v2_proxy_temporal_selector_metrics_step35 import (
    aggregate_setting_metrics,
    build_step35_gate_decision,
    combined_temporal_selector_loss,
    compute_temporal_prediction_metrics,
)
from data.bridgedata_v2_proxy_temporal_selector_splits_step35 import (
    build_step35_training_settings,
    summarize_step35_leakage,
)
from models.bridgedata_v2_proxy_temporal_selector import ProxyTemporalSelectorHead


def train_step35_proxy_temporal_selector(config_path: str | Path) -> dict[str, Any]:
    config = load_step35_config(config_path)
    env_guard = _env_guard()
    missing = missing_step35_inputs(config)
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        payload = _safe_stop_payload(config, "env_isaaclab unexpectedly contains TensorFlow/TFDS", env_guard)
        _write_outputs(config, payload)
        return payload
    if missing:
        payload = _safe_stop_payload(config, f"missing Step35 inputs: {missing}", env_guard)
        _write_outputs(config, payload)
        return payload

    settings = build_step35_training_settings(config)
    required_ids = sample_ids_by_shard_from_splits(config)
    samples_payload = load_step35_samples(config, required_ids)
    samples_by_shard = samples_payload["samples_by_shard"]
    leakage = summarize_step35_leakage(settings)
    device = _resolve_device(
        str(config["training"].get("device", "cpu")),
        allow_cpu_fallback=bool(config["training"].get("allow_cpu_fallback", True)),
    )
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    optimizer_step_performed = False
    for setting in settings:
        row, curve = _train_one_setting(config, setting, samples_by_shard, device)
        rows.append(row)
        curves.append(curve)
        optimizer_step_performed = optimizer_step_performed or bool(row.get("optimizer_step_performed", False))

    within = aggregate_setting_metrics([row for row in rows if row["eval_type"] == "within_shard"], eval_type="within_shard")
    cross = aggregate_setting_metrics([row for row in rows if row["eval_type"] == "cross_shard"], eval_type="cross_shard")
    mixed = aggregate_setting_metrics([row for row in rows if row["eval_type"] == "mixed_shard"], eval_type="mixed_shard")
    step34_plan = _read_json(config["input"]["step34_plan_summary_json"])
    evidence = step34_plan.get("step33b_evidence", {})
    dataset_bias_detected = bool(evidence.get("dataset_bias_detected", False))
    full_context_noise_ack = bool(evidence.get("full_context_noise_confirmed", False))
    gate = build_step35_gate_decision(
        within=within,
        cross=cross,
        mixed=mixed,
        leakage=leakage,
        dataset_bias_detected=dataset_bias_detected,
        full_context_noise_acknowledged=full_context_noise_ack,
    )
    train_summary = {
        "stage": STAGE,
        "safe_stop": False,
        "reason": None,
        "training_performed": True,
        "bounded_smoke_training_only": True,
        "temporal_selector_head_training_performed": True,
        "optimizer_step_performed": bool(optimizer_step_performed),
        "optimizer_scope": OPTIMIZER_SCOPE,
        "videomae_loaded": False,
        "videomae_training_performed": False,
        "videomae_frozen": True,
        "current_importance_training_performed": False,
        "final_selector_training_performed": False,
        "patch_level_selector_training_performed": False,
        "world_model_training_performed": False,
        "downstream_task_training_performed": False,
        "action_conditioned_world_model_training_performed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "new_dataset_download_performed": False,
        "model_download_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "num_settings": len(settings),
        "num_completed_settings": sum(not bool(row.get("skipped", False)) for row in rows),
        "sample_summary": samples_payload["summary"],
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": True,
    }
    metrics = {
        "stage": STAGE,
        "setting_rows": rows,
        "within_shard": within,
        "cross_shard": cross,
        "mixed_shard": mixed,
        "selector_beats_random_baseline": bool(gate["selector_beats_random_baseline"]),
        "selector_beats_current_only_baseline": bool(gate["selector_beats_current_only_baseline"]),
        "selector_generalizes_cross_shard": bool(gate["selector_generalizes_cross_shard"]),
        "mixed_shard_pass": bool(gate["mixed_shard_pass"]),
        "leakage_checks": leakage,
        "dataset_bias_detected": bool(dataset_bias_detected),
        "full_context_noise_acknowledged": bool(full_context_noise_ack),
        "safety_gate_pass": True,
    }
    payload = {
        "train_summary": train_summary,
        "loss_curves": {"stage": STAGE, "curves": curves, "num_curves": len(curves)},
        "selector_metrics": metrics,
        "gate_decision": gate,
    }
    _write_outputs(config, payload)
    return payload


def _train_one_setting(
    config: dict[str, Any],
    setting: dict[str, Any],
    samples_by_shard: dict[str, list[dict[str, Any]]],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_samples = _samples_for(setting["train_sample_ids_by_shard"], samples_by_shard)
    val_samples = _samples_for(setting["val_sample_ids_by_shard"], samples_by_shard)
    if not train_samples or not val_samples:
        reason = f"missing train/val samples for {setting['name']}"
        row = {
            "setting": setting["name"],
            "eval_type": setting["eval_type"],
            "split_seed": setting["split_seed"],
            "skipped": True,
            "skip_reason": reason,
            "optimizer_step_performed": False,
            "safety_gate_pass": True,
        }
        return row, {"setting": setting["name"], "skipped": True, "skip_reason": reason, "curve": []}

    _seed_everything(int(config.get("seed", 42)) + int(setting["split_seed"]))
    selector_cfg = config["selector"]
    model = ProxyTemporalSelectorHead(
        token_dim=int(selector_cfg.get("token_dim", 768)),
        hidden_dim=int(selector_cfg.get("hidden_dim", 128)),
        context_frames=int(selector_cfg.get("context_frames", 16)),
        spatial_tokens=int(selector_cfg.get("spatial_tokens", 392)),
        condition_on_current_summary=bool(selector_cfg.get("condition_on_current_summary", True)),
        dropout=float(selector_cfg.get("dropout", 0.0)),
        detach_token_inputs=bool(selector_cfg.get("detach_token_inputs", True)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 0.001)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scope = _optimizer_scope(model, optimizer)
    if scope["optimizer_scope"] != OPTIMIZER_SCOPE:
        raise RuntimeError(f"optimizer scope must be {OPTIMIZER_SCOPE}: {scope}")

    train_steps = int(config["training"].get("train_steps", 300))
    eval_every = int(config["training"].get("eval_every", 25))
    batch_size = int(config["training"].get("batch_size", 8))
    grad_clip = float(config["training"].get("grad_clip_norm", 1.0))
    rng = random.Random(int(config.get("seed", 42)) + int(setting["split_seed"]))
    curve: list[dict[str, Any]] = []
    optimizer_step_performed = False
    for step in range(1, train_steps + 1):
        batch_samples = [train_samples[rng.randrange(len(train_samples))] for _ in range(min(batch_size, len(train_samples)))]
        batch = batch_step35_samples(batch_samples, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_temporal_features(
            batch["context_temporal_features"],
            current_summary=batch["current_summary"],
        )
        loss_parts = combined_temporal_selector_loss(logits, batch["proxy_temporal_target"], config["loss"])
        loss_parts["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer_step_performed = True
        if step == 1 or step % eval_every == 0 or step == train_steps:
            train_eval = _evaluate_samples(model, train_samples, device, random_seed=setting["split_seed"] + step)
            val_eval = _evaluate_samples(model, val_samples, device, random_seed=setting["split_seed"] + step + 1000)
            curve.append(
                {
                    "step": step,
                    "train_mse": train_eval["selector_val_mse"],
                    "val_mse": val_eval["selector_val_mse"],
                    "mse_loss": float(loss_parts["mse_loss"].item()),
                    "rank_loss": float(loss_parts["rank_loss"].item()),
                    "topk_soft_loss": float(loss_parts["topk_soft_loss"].item()),
                }
            )

    final_train = _evaluate_samples(model, train_samples, device, random_seed=setting["split_seed"] + 3000)
    final_val = _evaluate_samples(model, val_samples, device, random_seed=setting["split_seed"] + 4000)
    row = {
        "setting": setting["name"],
        "eval_type": setting["eval_type"],
        "split_seed": setting["split_seed"],
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "val_shards": sorted(setting["val_sample_ids_by_shard"].keys()),
        "selector_val_mse": final_val["selector_val_mse"],
        "random_baseline_mse": final_val["random_baseline_mse"],
        "current_only_baseline_mse": final_val["current_only_baseline_mse"],
        "uniform_baseline_mse": final_val["uniform_baseline_mse"],
        "selector_beats_random_baseline": final_val["selector_beats_random_baseline"],
        "selector_beats_current_only_baseline": final_val["selector_beats_current_only_baseline"],
        "selector_beats_uniform_baseline": final_val["selector_beats_uniform_baseline"],
        "selector_spearman": final_val["selector_spearman"],
        "selector_pearson": final_val["selector_pearson"],
        "top1_frame_hit": final_val["top1_frame_hit"],
        "top2_frame_overlap": final_val["top2_frame_overlap"],
        "top4_frame_overlap": final_val["top4_frame_overlap"],
        "val_mse": final_val["val_mse"],
        "val_spearman": final_val["val_spearman"],
        "val_pearson": final_val["val_pearson"],
        "train_mse": final_train["selector_val_mse"],
        "overfit_gap": float(final_val["selector_val_mse"] - final_train["selector_val_mse"]),
        "optimizer_step_performed": bool(optimizer_step_performed),
        "optimizer_scope": OPTIMIZER_SCOPE,
        "optimizer_param_count": scope["optimizer_param_count"],
        "videomae_training_performed": False,
        "current_importance_training_performed": False,
        "final_selector_training_performed": False,
        "patch_level_selector_training_performed": False,
        "checkpoint_saved": False,
        "safety_gate_pass": True,
    }
    return row, {
        "setting": setting["name"],
        "eval_type": setting["eval_type"],
        "split_seed": setting["split_seed"],
        "curve": curve,
    }


def _evaluate_samples(
    model: ProxyTemporalSelectorHead,
    samples: list[dict[str, Any]],
    device: torch.device,
    *,
    random_seed: int,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        batch = batch_step35_samples(samples, device)
        logits = model.forward_temporal_features(
            batch["context_temporal_features"],
            current_summary=batch["current_summary"],
        )
        pred = torch.sigmoid(logits)
    return compute_temporal_prediction_metrics(
        pred.detach().cpu(),
        batch["proxy_temporal_target"].detach().cpu(),
        random_seed=random_seed,
    )


def _samples_for(ids_by_shard: dict[str, list[str]], samples_by_shard: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for shard, sample_ids in ids_by_shard.items():
        by_id = {str(sample["sample_id"]): sample for sample in samples_by_shard.get(str(shard), [])}
        result.extend(by_id[str(sample_id)] for sample_id in sample_ids if str(sample_id) in by_id)
    return result


def _optimizer_scope(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    model_param_ids = {id(param) for param in model.parameters() if param.requires_grad}
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group.get("params", [])
        if getattr(param, "requires_grad", False)
    }
    return {
        "optimizer_scope": OPTIMIZER_SCOPE if optimizer_param_ids == model_param_ids and optimizer_param_ids else "invalid",
        "optimizer_param_count": sum(int(param.numel()) for param in model.parameters() if param.requires_grad),
        "optimizer_matches_selector_head_only": optimizer_param_ids == model_param_ids and bool(optimizer_param_ids),
    }


def _safe_stop_payload(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    summary = {
        "stage": STAGE,
        "safe_stop": True,
        "reason": reason,
        "training_performed": False,
        "bounded_smoke_training_only": True,
        "temporal_selector_head_training_performed": False,
        "optimizer_step_performed": False,
        "optimizer_scope": OPTIMIZER_SCOPE,
        "videomae_training_performed": False,
        "videomae_frozen": True,
        "current_importance_training_performed": False,
        "final_selector_training_performed": False,
        "patch_level_selector_training_performed": False,
        "checkpoint_saved": False,
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }
    empty = {
        "stage": STAGE,
        "setting_rows": [],
        "within_shard": {"eval_type": "within_shard", "num_rows": 0, "skipped": True},
        "cross_shard": {"eval_type": "cross_shard", "num_rows": 0, "skipped": True},
        "mixed_shard": {"eval_type": "mixed_shard", "num_rows": 0, "skipped": True},
        "selector_beats_random_baseline": False,
        "selector_beats_current_only_baseline": False,
        "selector_generalizes_cross_shard": False,
        "safety_gate_pass": summary["safety_gate_pass"],
    }
    gate = {
        "stage": STAGE,
        "future_selector_training_gate_ready": False,
        "selector_training_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "reason": reason,
        "recommended_step36": {
            "name": "restore Step33B/Step34 inputs before Step35",
            "scope": "no downloads or final selector training",
        },
        "safety_gate_pass": summary["safety_gate_pass"],
    }
    return {
        "train_summary": summary,
        "loss_curves": {"stage": STAGE, "curves": [], "num_curves": 0},
        "selector_metrics": empty,
        "gate_decision": gate,
    }


def _write_outputs(config: dict[str, Any], payload: dict[str, Any]) -> None:
    _write_json(config["output"]["train_summary_json"], payload["train_summary"])
    _write_json(config["output"]["loss_curves_json"], payload["loss_curves"])
    _write_json(config["output"]["metrics_json"], payload["selector_metrics"])
    _write_json(config["output"]["gate_decision_json"], payload["gate_decision"])


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_device(device: str, *, allow_cpu_fallback: bool) -> torch.device:
    if device == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        if allow_cpu_fallback:
            return torch.device("cpu")
        raise RuntimeError("cuda requested but unavailable")
    return torch.device(device)


def _env_guard() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
