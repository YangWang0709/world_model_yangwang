"""Bounded Step41A current-conditioning selector diagnosis trainer."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import torch

from data.bridgedata_v2_current_conditioning_dataset_step41a import (
    OPTIMIZER_SCOPE,
    STAGE,
    attach_step41a_targets,
    batch_step41a_samples,
    build_step41a_training_settings,
    build_train_split_global_spatial_prior,
    load_step41a_config,
    load_step41a_samples,
    missing_step41a_inputs,
)
from data.bridgedata_v2_current_conditioning_losses_step41a import current_conditioning_mse_loss
from data.bridgedata_v2_current_conditioning_metrics_step41a import (
    add_current_gain_metrics,
    build_step41a_gate_decision,
    build_variant_comparison,
    compute_current_conditioning_metrics,
)
from data.bridgedata_v2_proxy_patch_selector_splits_step36 import summarize_step36_leakage
from models.bridgedata_v2_current_conditioned_selector_step41a import (
    VARIANT_NAMES,
    CurrentConditionedSelectorHead,
    build_current_conditioned_selector,
)


def train_step41a_current_conditioning(config_path: str | Path) -> dict[str, Any]:
    config = load_step41a_config(config_path)
    env_guard = _env_guard()
    missing = missing_step41a_inputs(config)
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        payload = _safe_stop_payload(config, "env_isaaclab unexpectedly contains TensorFlow/TFDS", env_guard)
        _write_outputs(config, payload)
        return payload
    if missing:
        payload = _safe_stop_payload(config, f"missing or unsafe Step41A inputs: {missing}", env_guard)
        _write_outputs(config, payload)
        return payload

    settings = build_step41a_training_settings(config)
    samples_payload = load_step41a_samples(config)
    samples_by_shard = samples_payload["samples_by_shard"]
    leakage = summarize_step36_leakage(settings)
    device = _resolve_device(
        str(config["training"].get("device", "cpu")),
        allow_cpu_fallback=bool(config["training"].get("allow_cpu_fallback", True)),
    )
    variant_names = _enabled_variants(config)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    optimizer_step_performed = False
    for setting in settings:
        for variant_name in variant_names:
            row, curve = _train_one_setting_variant(config, setting, samples_by_shard, device, variant_name)
            rows.append(row)
            curves.append(curve)
            optimizer_step_performed = optimizer_step_performed or bool(row.get("optimizer_step_performed", False))

    add_current_gain_metrics(rows)
    variant_comparison = build_variant_comparison(rows, variant_names)
    gate = build_step41a_gate_decision(
        variant_comparison=variant_comparison,
        leakage=leakage,
        requirements=config.get("diagnostic_gate_requirements", {}),
    )
    best_variant = gate.get("best_variant")
    train_summary = {
        "stage": STAGE,
        "safe_stop": False,
        "reason": None,
        "training_performed": True,
        "bounded_current_conditioning_smoke_training_performed": True,
        "bounded_current_conditioning_smoke_training_only": True,
        "variant_screening_only": True,
        "optimizer_step_performed": bool(optimizer_step_performed),
        "optimizer_scope": OPTIMIZER_SCOPE,
        "variants_run": variant_names,
        "best_variant": best_variant,
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
        "future_tokens_exposed_to_selector": False,
        "future_tokens_used_as_input": False,
        "label_variant": config["label"]["variant"],
        "num_settings": len(settings),
        "num_variants": len(variant_names),
        "num_completed_setting_variants": sum(not bool(row.get("skipped", False)) for row in rows),
        "sample_summary": samples_payload["summary"],
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": True,
    }
    metrics = {
        "stage": STAGE,
        "setting_variant_rows": rows,
        "variant_names": variant_names,
        "best_variant": best_variant,
        "best_variant_overall": _best_metric(variant_comparison, best_variant, "overall"),
        "best_variant_within_shard": _best_metric(variant_comparison, best_variant, "within_shard"),
        "best_variant_cross_shard": _best_metric(variant_comparison, best_variant, "cross_shard"),
        "best_variant_mixed_shard": _best_metric(variant_comparison, best_variant, "mixed_shard"),
        "leakage_checks": leakage,
        "label_variant": config["label"]["variant"],
        "safety_gate_pass": True,
    }
    payload = {
        "train_summary": train_summary,
        "loss_curves": {"stage": STAGE, "curves": curves, "num_curves": len(curves)},
        "current_conditioning_metrics": metrics,
        "variant_comparison": variant_comparison,
        "gate_decision": gate,
    }
    _write_outputs(config, payload)
    return payload


def _train_one_setting_variant(
    config: dict[str, Any],
    setting: dict[str, Any],
    samples_by_shard: dict[str, list[dict[str, Any]]],
    device: torch.device,
    variant_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_raw = _samples_for(setting["train_sample_ids_by_shard"], samples_by_shard)
    val_raw = _samples_for(setting["val_sample_ids_by_shard"], samples_by_shard)
    if not train_raw or not val_raw:
        reason = f"missing train/val samples for {setting['name']} {variant_name}"
        row = {
            "setting": setting["name"],
            "eval_type": setting["eval_type"],
            "split_seed": setting["split_seed"],
            "variant_name": variant_name,
            "skipped": True,
            "skip_reason": reason,
            "optimizer_step_performed": False,
            "safety_gate_pass": True,
        }
        return row, {"setting": setting["name"], "variant_name": variant_name, "skipped": True, "skip_reason": reason}

    prior_payload = build_train_split_global_spatial_prior(train_raw)
    train_prior = prior_payload["prior"]
    train_samples = attach_step41a_targets(train_raw, train_prior)
    val_samples = attach_step41a_targets(val_raw, train_prior)
    seed = int(config.get("seed", 42)) + int(setting["split_seed"]) + _variant_offset(variant_name)
    _seed_everything(seed)
    model = build_current_conditioned_selector(config, variant_name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 0.001)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scope = _optimizer_scope(model, optimizer)
    if scope["optimizer_scope"] != OPTIMIZER_SCOPE:
        raise RuntimeError(f"optimizer scope must be {OPTIMIZER_SCOPE}: {scope}")

    train_steps = int(config["training"].get("train_steps", 200))
    eval_every = int(config["training"].get("eval_every", 25))
    batch_size = int(config["training"].get("batch_size", 2))
    eval_batch_size = int(config["training"].get("eval_batch_size", batch_size))
    grad_clip = float(config["training"].get("grad_clip_norm", 1.0))
    topk_values = [int(k) for k in config["metrics"].get("topk_values", [64, 128, 256, 512])]
    max_attention = int(config["current_conditioning_variants"].get("max_attention_elements_per_batch", 20000000))
    rng = random.Random(seed)
    curve: list[dict[str, Any]] = []
    optimizer_step_performed = False
    max_attention_seen = 0
    for step in range(1, train_steps + 1):
        batch_samples = [train_samples[rng.randrange(len(train_samples))] for _ in range(min(batch_size, len(train_samples)))]
        batch = batch_step41a_samples(batch_samples, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["context_tokens"], batch["current_tokens"])
        max_attention_seen = max(max_attention_seen, int(output.get("diagnostics", {}).get("max_attention_elements_seen", 0)))
        if max_attention_seen > max_attention:
            raise RuntimeError(f"Step41A attention element guard exceeded: {max_attention_seen} > {max_attention}")
        loss_parts = current_conditioning_mse_loss(output, batch["target_label"], config["loss"])
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
                random_seed=seed + step,
                topk_values=topk_values,
                batch_size=eval_batch_size,
            )
            val_eval = _evaluate_samples(
                model,
                val_samples,
                train_prior,
                device,
                random_seed=seed + step + 1000,
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
        random_seed=seed + 3000,
        topk_values=topk_values,
        batch_size=eval_batch_size,
    )
    final_val = _evaluate_samples(
        model,
        val_samples,
        train_prior,
        device,
        random_seed=seed + 4000,
        topk_values=topk_values,
        batch_size=eval_batch_size,
    )
    row = {
        "setting": setting["name"],
        "eval_type": setting["eval_type"],
        "split_seed": setting["split_seed"],
        "variant_name": variant_name,
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
        "top256_overlap": final_val.get("top256_overlap", final_val["top256_precision"]),
        "top512_overlap": final_val.get("top512_overlap", 0.0),
        "top256_precision": final_val["top256_precision"],
        "top256_recall": final_val["top256_recall"],
        "train_mse": final_train["selector_val_mse"],
        "overfit_gap": float(final_val["selector_val_mse"] - final_train["selector_val_mse"]),
        "all_finite": bool(final_val["all_finite"]),
        "max_attention_elements_seen": int(max_attention_seen),
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
    return (
        row,
        {
            "setting": setting["name"],
            "eval_type": setting["eval_type"],
            "split_seed": setting["split_seed"],
            "variant_name": variant_name,
            "curve": curve,
        },
    )


def _evaluate_samples(
    model: CurrentConditionedSelectorHead,
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
            batch = batch_step41a_samples(chunk, device)
            output = model(batch["context_tokens"], batch["current_tokens"])
            preds.append(torch.sigmoid(output["scores"]).detach().cpu())
            targets.append(batch["target_label"].detach().cpu())
    return compute_current_conditioning_metrics(
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
    model_param_ids = {id(param) for param in model.parameters() if param.requires_grad}
    optim_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    return {
        "optimizer_scope": OPTIMIZER_SCOPE if optim_param_ids == model_param_ids else "unexpected_optimizer_scope",
        "optimizer_param_count": sum(param.numel() for group in optimizer.param_groups for param in group["params"]),
        "model_param_count": sum(param.numel() for param in model.parameters() if param.requires_grad),
    }


def _safe_stop_payload(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    summary = {
        "stage": STAGE,
        "safe_stop": True,
        "reason": reason,
        "training_performed": False,
        "bounded_current_conditioning_smoke_training_performed": False,
        "optimizer_step_performed": False,
        "optimizer_scope": OPTIMIZER_SCOPE,
        "variants_run": [],
        "best_variant": None,
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
        "future_tokens_exposed_to_selector": False,
        "future_tokens_used_as_input": False,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }
    empty_metric = {
        "stage": STAGE,
        "setting_variant_rows": [],
        "variant_names": [],
        "best_variant": None,
        "safety_gate_pass": summary["safety_gate_pass"],
    }
    comparison = {
        "stage": STAGE,
        "variant_names": [],
        "per_variant": {},
        "ranked_variants": [],
        "best_variant": None,
        "safety_gate_pass": summary["safety_gate_pass"],
    }
    gate = {
        "stage": STAGE,
        "current_conditioning_candidate_ready": False,
        "best_variant": None,
        "best_variant_beats_no_current": False,
        "best_variant_beats_current_mean_summary": False,
        "best_variant_beats_random_token_baseline": False,
        "best_variant_beats_uniform_or_mean_baseline": False,
        "best_variant_beats_temporal_broadcast_baseline": False,
        "best_variant_generalizes_cross_shard": False,
        "top256_overlap_mean": 0.0,
        "overfit_gap_mean_abs": 0.0,
        "bounded_current_conditioning_smoke_performed": False,
        "downstream_selector_use_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "reason": reason,
        "recommended_step42": {"name": "restore Step41A inputs", "scope": "no downloads"},
        "safety_gate_pass": summary["safety_gate_pass"],
    }
    return {
        "train_summary": summary,
        "loss_curves": {"stage": STAGE, "curves": [], "num_curves": 0},
        "current_conditioning_metrics": empty_metric,
        "variant_comparison": comparison,
        "gate_decision": gate,
    }


def _write_outputs(config: dict[str, Any], payload: dict[str, Any]) -> None:
    _write_json(config["output"]["train_summary_json"], payload["train_summary"])
    _write_json(config["output"]["loss_curves_json"], payload["loss_curves"])
    _write_json(config["output"]["current_conditioning_metrics_json"], payload["current_conditioning_metrics"])
    _write_json(config["output"]["variant_comparison_json"], payload["variant_comparison"])
    _write_json(config["output"]["gate_decision_json"], payload["gate_decision"])


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _best_metric(variant_comparison: dict[str, Any], best_variant: str | None, key: str) -> dict[str, Any]:
    if not best_variant:
        return {}
    return variant_comparison.get("per_variant", {}).get(best_variant, {}).get(key, {})


def _enabled_variants(config: dict[str, Any]) -> list[str]:
    variants = [str(name) for name in config.get("current_conditioning_variants", {}).get("enabled", VARIANT_NAMES)]
    unknown = [name for name in variants if name not in VARIANT_NAMES]
    if unknown:
        raise ValueError(f"unknown Step41A variants: {unknown}")
    return variants


def _variant_offset(variant_name: str) -> int:
    return {name: index * 10000 for index, name in enumerate(VARIANT_NAMES)}[variant_name]


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
