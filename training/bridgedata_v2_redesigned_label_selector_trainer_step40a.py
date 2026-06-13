"""Bounded Step40A redesigned-label selector smoke trainer."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import torch

from data.bridgedata_v2_proxy_patch_selector_splits_step36 import summarize_step36_leakage
from data.bridgedata_v2_redesigned_label_selector_dataset_step40a import (
    OPTIMIZER_SCOPE,
    STAGE,
    attach_targets,
    batch_step40a_samples,
    build_step40a_training_settings,
    build_train_split_global_spatial_prior,
    load_step40a_config,
    load_step40a_samples,
    missing_step40a_inputs,
)
from data.bridgedata_v2_redesigned_label_selector_losses_step40a import redesigned_label_selector_mse_loss
from data.bridgedata_v2_redesigned_label_selector_metrics_step40a import (
    aggregate_redesigned_label_selector_metrics,
    build_step40a_gate_decision,
    compute_redesigned_label_selector_metrics,
)
from models.bridgedata_v2_proxy_patch_token_selector import ProxyPatchTokenSelectorHead


def train_step40a_redesigned_label_selector(config_path: str | Path) -> dict[str, Any]:
    config = load_step40a_config(config_path)
    env_guard = _env_guard()
    missing = missing_step40a_inputs(config)
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        payload = _safe_stop_payload(config, "env_isaaclab unexpectedly contains TensorFlow/TFDS", env_guard)
        _write_outputs(config, payload)
        return payload
    if missing:
        payload = _safe_stop_payload(config, f"missing Step40A inputs: {missing}", env_guard)
        _write_outputs(config, payload)
        return payload

    settings = build_step40a_training_settings(config)
    samples_payload = load_step40a_samples(config)
    samples_by_shard = samples_payload["samples_by_shard"]
    leakage = summarize_step36_leakage(settings)
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

    within = aggregate_redesigned_label_selector_metrics(
        [row for row in rows if row["eval_type"] == "within_shard"], eval_type="within_shard"
    )
    cross = aggregate_redesigned_label_selector_metrics(
        [row for row in rows if row["eval_type"] == "cross_shard"], eval_type="cross_shard"
    )
    mixed = aggregate_redesigned_label_selector_metrics(
        [row for row in rows if row["eval_type"] == "mixed_shard"], eval_type="mixed_shard"
    )
    gate = build_step40a_gate_decision(
        within=within,
        cross=cross,
        mixed=mixed,
        leakage=leakage,
        requirements=config.get("gate_requirements", {}),
    )
    train_summary = {
        "stage": STAGE,
        "safe_stop": False,
        "reason": None,
        "training_performed": True,
        "bounded_selector_smoke_training_performed": True,
        "bounded_smoke_training_only": True,
        "patch_token_selector_head_training_performed": True,
        "optimizer_step_performed": bool(optimizer_step_performed),
        "optimizer_scope": OPTIMIZER_SCOPE,
        "videomae_loaded": False,
        "videomae_training_performed": False,
        "videomae_frozen": True,
        "current_importance_training_performed": False,
        "final_selector_training_performed": False,
        "world_model_training_performed": False,
        "downstream_task_training_performed": False,
        "action_conditioned_world_model_training_performed": False,
        "downstream_selector_use_allowed": False,
        "context_utility_claim_allowed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "new_dataset_download_performed": False,
        "model_download_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "future_tokens_used_as_input": False,
        "label_variant": config["label"]["variant"],
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
        "selector_beats_random_token_baseline": bool(gate["selector_beats_random_token_baseline"]),
        "selector_beats_uniform_or_mean_baseline": bool(gate["selector_beats_uniform_or_mean_baseline"]),
        "selector_beats_temporal_broadcast_baseline": bool(gate["selector_beats_temporal_broadcast_baseline"]),
        "selector_beats_train_global_spatial_prior_baseline": bool(
            gate["selector_beats_train_global_spatial_prior_baseline"]
        ),
        "selector_generalizes_cross_shard": bool(gate["selector_generalizes_cross_shard"]),
        "top256_overlap_mean": float(gate["top256_overlap_mean"]),
        "overfit_gap_mean_abs": float(gate["overfit_gap_mean_abs"]),
        "leakage_checks": leakage,
        "label_variant": config["label"]["variant"],
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
    train_raw = _samples_for(setting["train_sample_ids_by_shard"], samples_by_shard)
    val_raw = _samples_for(setting["val_sample_ids_by_shard"], samples_by_shard)
    if not train_raw or not val_raw:
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

    prior_payload = build_train_split_global_spatial_prior(train_raw)
    train_prior = prior_payload["prior"]
    train_samples = attach_targets(train_raw, train_prior)
    val_samples = attach_targets(val_raw, train_prior)
    _seed_everything(int(config.get("seed", 42)) + int(setting["split_seed"]))
    model_cfg = config["model"]
    model = ProxyPatchTokenSelectorHead(
        token_dim=int(model_cfg.get("token_dim", 768)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        context_frames=int(model_cfg.get("context_frames", 16)),
        spatial_tokens=int(model_cfg.get("spatial_tokens", 392)),
        condition_on_current_summary=bool(model_cfg.get("condition_on_current_summary", True)),
        use_temporal_embedding=bool(model_cfg.get("use_temporal_embedding", True)),
        use_spatial_embedding=bool(model_cfg.get("use_spatial_embedding", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        detach_token_inputs=bool(model_cfg.get("detach_token_inputs", True)),
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
    batch_size = int(config["training"].get("batch_size", 4))
    eval_batch_size = int(config["training"].get("eval_batch_size", batch_size))
    grad_clip = float(config["training"].get("grad_clip_norm", 1.0))
    topk_values = [int(k) for k in config["metrics"].get("topk_values", [64, 128, 256, 512])]
    rng = random.Random(int(config.get("seed", 42)) + int(setting["split_seed"]))
    curve: list[dict[str, Any]] = []
    optimizer_step_performed = False
    for step in range(1, train_steps + 1):
        batch_samples = [train_samples[rng.randrange(len(train_samples))] for _ in range(min(batch_size, len(train_samples)))]
        batch = batch_step40a_samples(batch_samples, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["context_tokens"], batch["current_tokens"])
        loss_parts = redesigned_label_selector_mse_loss(logits, batch["target_label"], config["loss"])
        loss_parts["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer_step_performed = True
        if step == 1 or step % eval_every == 0 or step == train_steps:
            train_eval = _evaluate_samples(
                model,
                train_samples,
                train_prior,
                device,
                random_seed=setting["split_seed"] + step,
                topk_values=topk_values,
                batch_size=eval_batch_size,
            )
            val_eval = _evaluate_samples(
                model,
                val_samples,
                train_prior,
                device,
                random_seed=setting["split_seed"] + step + 1000,
                topk_values=topk_values,
                batch_size=eval_batch_size,
            )
            curve.append(
                {
                    "step": step,
                    "train_mse": train_eval["selector_val_mse"],
                    "val_mse": val_eval["selector_val_mse"],
                    "mse_loss": float(loss_parts["mse_loss"].item()),
                }
            )

    final_train = _evaluate_samples(
        model,
        train_samples,
        train_prior,
        device,
        random_seed=setting["split_seed"] + 3000,
        topk_values=topk_values,
        batch_size=eval_batch_size,
    )
    final_val = _evaluate_samples(
        model,
        val_samples,
        train_prior,
        device,
        random_seed=setting["split_seed"] + 4000,
        topk_values=topk_values,
        batch_size=eval_batch_size,
    )
    row = {
        "setting": setting["name"],
        "eval_type": setting["eval_type"],
        "split_seed": setting["split_seed"],
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "val_shards": sorted(setting["val_sample_ids_by_shard"].keys()),
        "label_variant": config["label"]["variant"],
        "train_global_spatial_prior_stats": prior_payload["stats"],
        "selector_val_mse": final_val["selector_val_mse"],
        "selector_val_mae": final_val["selector_val_mae"],
        "random_token_baseline_mse": final_val["random_token_baseline_mse"],
        "uniform_or_mean_baseline_mse": final_val["uniform_or_mean_baseline_mse"],
        "temporal_broadcast_baseline_mse": final_val["temporal_broadcast_baseline_mse"],
        "train_global_spatial_prior_baseline_mse": final_val["train_global_spatial_prior_baseline_mse"],
        "selector_beats_random_token_baseline": final_val["selector_beats_random_token_baseline"],
        "selector_beats_uniform_or_mean_baseline": final_val["selector_beats_uniform_or_mean_baseline"],
        "selector_beats_temporal_broadcast_baseline": final_val["selector_beats_temporal_broadcast_baseline"],
        "selector_beats_train_global_spatial_prior_baseline": final_val[
            "selector_beats_train_global_spatial_prior_baseline"
        ],
        "pearson": final_val["pearson"],
        "spearman": final_val["spearman"],
        "top64_overlap": final_val.get("top64_overlap", 0.0),
        "top128_overlap": final_val.get("top128_overlap", 0.0),
        "top256_overlap": final_val["top256_overlap"],
        "top512_overlap": final_val.get("top512_overlap", 0.0),
        "top256_precision": final_val["top256_precision"],
        "top256_recall": final_val["top256_recall"],
        "train_mse": final_train["selector_val_mse"],
        "overfit_gap": float(final_val["selector_val_mse"] - final_train["selector_val_mse"]),
        "all_finite": bool(final_val["all_finite"]),
        "optimizer_step_performed": bool(optimizer_step_performed),
        "optimizer_scope": OPTIMIZER_SCOPE,
        "optimizer_param_count": scope["optimizer_param_count"],
        "videomae_loaded": False,
        "videomae_training_performed": False,
        "current_importance_training_performed": False,
        "final_selector_training_performed": False,
        "world_model_training_performed": False,
        "downstream_task_training_performed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "safety_gate_pass": True,
    }
    return row, {"setting": setting["name"], "eval_type": setting["eval_type"], "split_seed": setting["split_seed"], "curve": curve}


def _evaluate_samples(
    model: ProxyPatchTokenSelectorHead,
    samples: list[dict[str, Any]],
    train_prior: torch.Tensor,
    device: torch.device,
    *,
    random_seed: int,
    topk_values: list[int],
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk in _chunks(samples, max(1, int(batch_size))):
            batch = batch_step40a_samples(chunk, device)
            logits = model(batch["context_tokens"], batch["current_tokens"])
            preds.append(torch.sigmoid(logits).detach().cpu())
            targets.append(batch["target_label"].detach().cpu())
    return compute_redesigned_label_selector_metrics(
        torch.cat(preds, dim=0),
        torch.cat(targets, dim=0),
        train_prior,
        random_seed=random_seed,
        topk_values=topk_values,
    )


def _samples_for(ids_by_shard: dict[str, list[str]], samples_by_shard: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for shard, sample_ids in ids_by_shard.items():
        by_id = {str(sample["sample_id"]): sample for sample in samples_by_shard.get(str(shard), [])}
        result.extend(by_id[str(sample_id)] for sample_id in sample_ids if str(sample_id) in by_id)
    return result


def _optimizer_scope(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    model_param_ids = {id(param) for param in model.parameters()}
    optim_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    return {
        "optimizer_scope": OPTIMIZER_SCOPE if optim_param_ids == model_param_ids else "unexpected_optimizer_scope",
        "optimizer_param_count": sum(param.numel() for group in optimizer.param_groups for param in group["params"]),
        "model_param_count": sum(param.numel() for param in model.parameters()),
    }


def _safe_stop_payload(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    summary = {
        "stage": STAGE,
        "safe_stop": True,
        "reason": reason,
        "training_performed": False,
        "bounded_selector_smoke_training_performed": False,
        "optimizer_step_performed": False,
        "optimizer_scope": OPTIMIZER_SCOPE,
        "videomae_loaded": False,
        "videomae_training_performed": False,
        "current_importance_training_performed": False,
        "final_selector_training_performed": False,
        "world_model_training_performed": False,
        "downstream_task_training_performed": False,
        "downstream_selector_use_allowed": False,
        "context_utility_claim_allowed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "new_dataset_download_performed": False,
        "model_download_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "future_tokens_used_as_input": False,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }
    empty_metric = {"eval_type": "none", "num_rows": 0, "skipped": True}
    metrics = {
        "stage": STAGE,
        "setting_rows": [],
        "within_shard": empty_metric,
        "cross_shard": empty_metric,
        "mixed_shard": empty_metric,
        "selector_beats_random_token_baseline": False,
        "selector_beats_uniform_or_mean_baseline": False,
        "selector_beats_temporal_broadcast_baseline": False,
        "selector_beats_train_global_spatial_prior_baseline": False,
        "selector_generalizes_cross_shard": False,
        "top256_overlap_mean": 0.0,
        "safety_gate_pass": summary["safety_gate_pass"],
    }
    gate = {
        "stage": STAGE,
        "redesigned_label_selector_smoke_pass": False,
        "bounded_selector_smoke_performed": False,
        "downstream_selector_use_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "reason": reason,
        "recommended_step41": {"name": "restore Step39A inputs before Step40A", "scope": "no downloads"},
        "safety_gate_pass": summary["safety_gate_pass"],
    }
    return {
        "train_summary": summary,
        "loss_curves": {"stage": STAGE, "curves": [], "num_curves": 0},
        "selector_metrics": metrics,
        "gate_decision": gate,
    }


def _write_outputs(config: dict[str, Any], payload: dict[str, Any]) -> None:
    _write_json(config["output"]["train_summary_json"], payload["train_summary"])
    _write_json(config["output"]["loss_curves_json"], payload["loss_curves"])
    _write_json(config["output"]["selector_metrics_json"], payload["selector_metrics"])
    _write_json(config["output"]["gate_decision_json"], payload["gate_decision"])


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def _chunks(samples: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [samples[index : index + size] for index in range(0, len(samples), size)]
