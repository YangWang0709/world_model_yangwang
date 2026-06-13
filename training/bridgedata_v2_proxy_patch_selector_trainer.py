"""Bounded Step36 proxy patch/token selector smoke trainer."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import torch

from data.bridgedata_v2_proxy_patch_selector_dataset_step36 import (
    OPTIMIZER_SCOPE,
    STAGE,
    batch_step36_samples,
    load_step36_config,
    load_step36_samples,
    missing_step36_inputs,
    sample_ids_by_shard_from_splits,
)
from data.bridgedata_v2_proxy_patch_selector_metrics_step36 import (
    aggregate_setting_metrics,
    build_step36_gate_decision,
    combined_patch_selector_loss,
    compute_patch_prediction_metrics,
)
from data.bridgedata_v2_proxy_patch_selector_splits_step36 import (
    build_step36_training_settings,
    summarize_step36_leakage,
)
from models.bridgedata_v2_proxy_patch_token_selector import ProxyPatchTokenSelectorHead


def train_step36_proxy_patch_selector(config_path: str | Path) -> dict[str, Any]:
    config = load_step36_config(config_path)
    env_guard = _env_guard()
    missing = missing_step36_inputs(config)
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        payload = _safe_stop_payload(config, "env_isaaclab unexpectedly contains TensorFlow/TFDS", env_guard)
        _write_outputs(config, payload)
        return payload
    if missing:
        payload = _safe_stop_payload(config, f"missing Step36 inputs: {missing}", env_guard)
        _write_outputs(config, payload)
        return payload

    settings = build_step36_training_settings(config)
    required_ids = sample_ids_by_shard_from_splits(config)
    samples_payload = load_step36_samples(config, required_ids)
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

    within = aggregate_setting_metrics([row for row in rows if row["eval_type"] == "within_shard"], eval_type="within_shard")
    cross = aggregate_setting_metrics([row for row in rows if row["eval_type"] == "cross_shard"], eval_type="cross_shard")
    mixed = aggregate_setting_metrics([row for row in rows if row["eval_type"] == "mixed_shard"], eval_type="mixed_shard")
    step35_gate = _read_json(config["input"]["step35_gate_decision_json"])
    dataset_bias_detected = bool(step35_gate.get("dataset_bias_detected", False))
    full_context_noise_ack = bool(step35_gate.get("full_context_noise_acknowledged", True))
    top256_min = float(config["future_gates"]["gate_requirements"].get("top256_overlap_mean_min", 0.10))
    gate = build_step36_gate_decision(
        within=within,
        cross=cross,
        mixed=mixed,
        leakage=leakage,
        dataset_bias_detected=dataset_bias_detected,
        full_context_noise_acknowledged=full_context_noise_ack,
        top256_overlap_mean_min=top256_min,
    )
    train_summary = {
        "stage": STAGE,
        "safe_stop": False,
        "reason": None,
        "training_performed": True,
        "bounded_smoke_training_only": True,
        "patch_token_selector_head_training_performed": True,
        "patch_level_selector_training_performed": True,
        "temporal_selector_head_training_performed": False,
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
        "patch_selector_beats_random_token_baseline": bool(gate["patch_selector_beats_random_token_baseline"]),
        "patch_selector_beats_uniform_token_baseline": bool(gate["patch_selector_beats_uniform_token_baseline"]),
        "patch_selector_beats_temporal_broadcast_baseline": bool(
            gate["patch_selector_beats_temporal_broadcast_baseline"]
        ),
        "patch_selector_beats_current_only_patch_baseline": bool(
            gate["patch_selector_beats_current_only_patch_baseline"]
        ),
        "patch_selector_generalizes_cross_shard": bool(gate["patch_selector_generalizes_cross_shard"]),
        "mixed_shard_pass": bool(gate["mixed_shard_pass"]),
        "top256_overlap_mean": float(gate["top256_overlap_mean"]),
        "leakage_checks": leakage,
        "dataset_bias_detected": bool(dataset_bias_detected),
        "full_context_noise_acknowledged": bool(full_context_noise_ack),
        "safety_gate_pass": True,
    }
    payload = {
        "train_summary": train_summary,
        "loss_curves": {"stage": STAGE, "curves": curves, "num_curves": len(curves)},
        "patch_selector_metrics": metrics,
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
    model = ProxyPatchTokenSelectorHead(
        token_dim=int(selector_cfg.get("token_dim", 768)),
        hidden_dim=int(selector_cfg.get("hidden_dim", 128)),
        context_frames=int(selector_cfg.get("context_frames", 16)),
        spatial_tokens=int(selector_cfg.get("spatial_tokens", 392)),
        condition_on_current_summary=bool(selector_cfg.get("condition_on_current_summary", True)),
        use_temporal_embedding=bool(selector_cfg.get("use_temporal_embedding", True)),
        use_spatial_embedding=bool(selector_cfg.get("use_spatial_embedding", True)),
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
    batch_size = int(config["training"].get("batch_size", 4))
    eval_batch_size = int(config["training"].get("eval_batch_size", batch_size))
    grad_clip = float(config["training"].get("grad_clip_norm", 1.0))
    topk_values = [int(k) for k in config["metrics"].get("topk_values", [64, 128, 256, 512])]
    rng = random.Random(int(config.get("seed", 42)) + int(setting["split_seed"]))
    curve: list[dict[str, Any]] = []
    optimizer_step_performed = False
    for step in range(1, train_steps + 1):
        batch_samples = [train_samples[rng.randrange(len(train_samples))] for _ in range(min(batch_size, len(train_samples)))]
        batch = batch_step36_samples(batch_samples, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["context_tokens"], batch["current_tokens"])
        loss_parts = combined_patch_selector_loss(logits, batch["proxy_patch_target"], config["loss"])
        loss_parts["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer_step_performed = True
        if step == 1 or step % eval_every == 0 or step == train_steps:
            train_eval = _evaluate_samples(
                model,
                train_samples,
                device,
                random_seed=setting["split_seed"] + step,
                topk_values=topk_values,
                batch_size=eval_batch_size,
            )
            val_eval = _evaluate_samples(
                model,
                val_samples,
                device,
                random_seed=setting["split_seed"] + step + 1000,
                topk_values=topk_values,
                batch_size=eval_batch_size,
            )
            curve.append(
                {
                    "step": step,
                    "train_mse": train_eval["patch_selector_val_mse"],
                    "val_mse": val_eval["patch_selector_val_mse"],
                    "mse_loss": float(loss_parts["mse_loss"].item()),
                    "rank_loss": float(loss_parts["rank_loss"].item()),
                    "topk_soft_loss": float(loss_parts["topk_soft_loss"].item()),
                }
            )

    final_train = _evaluate_samples(
        model,
        train_samples,
        device,
        random_seed=setting["split_seed"] + 3000,
        topk_values=topk_values,
        batch_size=eval_batch_size,
    )
    final_val = _evaluate_samples(
        model,
        val_samples,
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
        "patch_selector_val_mse": final_val["patch_selector_val_mse"],
        "random_token_baseline_mse": final_val["random_token_baseline_mse"],
        "uniform_token_baseline_mse": final_val["uniform_token_baseline_mse"],
        "temporal_broadcast_baseline_mse": final_val["temporal_broadcast_baseline_mse"],
        "current_only_patch_baseline_mse": final_val["current_only_patch_baseline_mse"],
        "patch_selector_beats_random_token_baseline": final_val["patch_selector_beats_random_token_baseline"],
        "patch_selector_beats_uniform_token_baseline": final_val["patch_selector_beats_uniform_token_baseline"],
        "patch_selector_beats_temporal_broadcast_baseline": final_val[
            "patch_selector_beats_temporal_broadcast_baseline"
        ],
        "patch_selector_beats_current_only_patch_baseline": final_val[
            "patch_selector_beats_current_only_patch_baseline"
        ],
        "patch_selector_spearman": final_val["patch_selector_spearman"],
        "patch_selector_pearson": final_val["patch_selector_pearson"],
        "top64_overlap": final_val["top64_overlap"],
        "top128_overlap": final_val["top128_overlap"],
        "top256_overlap": final_val["top256_overlap"],
        "top512_overlap": final_val["top512_overlap"],
        "top256_precision": final_val["top256_precision"],
        "top256_recall": final_val["top256_recall"],
        "val_mse": final_val["val_mse"],
        "val_spearman": final_val["val_spearman"],
        "val_pearson": final_val["val_pearson"],
        "train_mse": final_train["patch_selector_val_mse"],
        "overfit_gap": float(final_val["patch_selector_val_mse"] - final_train["patch_selector_val_mse"]),
        "optimizer_step_performed": bool(optimizer_step_performed),
        "optimizer_scope": OPTIMIZER_SCOPE,
        "optimizer_param_count": scope["optimizer_param_count"],
        "videomae_training_performed": False,
        "current_importance_training_performed": False,
        "final_selector_training_performed": False,
        "patch_level_selector_training_performed": True,
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
    model: ProxyPatchTokenSelectorHead,
    samples: list[dict[str, Any]],
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
            batch = batch_step36_samples(chunk, device)
            logits = model(batch["context_tokens"], batch["current_tokens"])
            preds.append(torch.sigmoid(logits).detach().cpu())
            targets.append(batch["proxy_patch_target"].detach().cpu())
    return compute_patch_prediction_metrics(
        torch.cat(preds, dim=0),
        torch.cat(targets, dim=0),
        random_seed=random_seed,
        topk_values=topk_values,
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
        "patch_token_selector_head_training_performed": False,
        "patch_level_selector_training_performed": False,
        "optimizer_step_performed": False,
        "optimizer_scope": OPTIMIZER_SCOPE,
        "videomae_training_performed": False,
        "videomae_frozen": True,
        "current_importance_training_performed": False,
        "final_selector_training_performed": False,
        "checkpoint_saved": False,
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "new_dataset_download_performed": False,
        "model_download_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }
    empty = {
        "stage": STAGE,
        "setting_rows": [],
        "within_shard": {"eval_type": "within_shard", "num_rows": 0, "skipped": True},
        "cross_shard": {"eval_type": "cross_shard", "num_rows": 0, "skipped": True},
        "mixed_shard": {"eval_type": "mixed_shard", "num_rows": 0, "skipped": True},
        "patch_selector_beats_random_token_baseline": False,
        "patch_selector_beats_uniform_token_baseline": False,
        "patch_selector_beats_temporal_broadcast_baseline": False,
        "patch_selector_generalizes_cross_shard": False,
        "top256_overlap_mean": 0.0,
        "safety_gate_pass": summary["safety_gate_pass"],
    }
    gate = {
        "stage": STAGE,
        "future_patch_selector_gate_ready": False,
        "patch_selector_beats_random_token_baseline": False,
        "patch_selector_beats_uniform_token_baseline": False,
        "patch_selector_beats_temporal_broadcast_baseline": False,
        "patch_selector_generalizes_cross_shard": False,
        "top256_overlap_mean": 0.0,
        "selector_training_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "reason": reason,
        "recommended_step37": {
            "name": "restore Step33B/Step35 inputs before Step36",
            "scope": "no downloads or final selector training",
        },
        "safety_gate_pass": summary["safety_gate_pass"],
    }
    return {
        "train_summary": summary,
        "loss_curves": {"stage": STAGE, "curves": [], "num_curves": 0},
        "patch_selector_metrics": empty,
        "gate_decision": gate,
    }


def _write_outputs(config: dict[str, Any], payload: dict[str, Any]) -> None:
    _write_json(config["output"]["train_summary_json"], payload["train_summary"])
    _write_json(config["output"]["loss_curves_json"], payload["loss_curves"])
    _write_json(config["output"]["metrics_json"], payload["patch_selector_metrics"])
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


def _chunks(samples: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [samples[index : index + size] for index in range(0, len(samples), size)]
