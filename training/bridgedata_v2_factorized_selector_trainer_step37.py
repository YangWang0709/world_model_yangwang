"""Bounded Step37 factorized selector diagnosis trainer."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import torch

from data.bridgedata_v2_factorized_selector_dataset_step37 import (
    ABLATION_OPTIMIZER_SCOPE,
    OPTIMIZER_SCOPE,
    STAGE,
    batch_step37_samples,
    build_step37_training_settings,
    load_step37_config,
    load_step37_samples,
    missing_step37_inputs,
    sample_ids_by_shard_from_splits,
)
from data.bridgedata_v2_factorized_selector_losses_step37 import (
    LOSS_VARIANTS,
    factorized_selector_loss,
)
from data.bridgedata_v2_factorized_selector_metrics_step37 import (
    aggregate_factorized_metrics,
    build_step37_gate_decision,
    compute_factorized_prediction_metrics,
)
from data.bridgedata_v2_proxy_patch_selector_splits_step36 import summarize_step36_leakage
from models.bridgedata_v2_proxy_factorized_selector import (
    ProxyFactorizedSelectorHead,
    ProxyFactorizedSelectorWithProxyTemporalPrior,
)


def train_step37_factorized_selector(config_path: str | Path) -> dict[str, Any]:
    config = load_step37_config(config_path)
    env_guard = _env_guard()
    missing = missing_step37_inputs(config)
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        payload = _safe_stop_payload(config, "env_isaaclab unexpectedly contains TensorFlow/TFDS", env_guard)
        _write_outputs(config, payload)
        return payload
    if missing:
        payload = _safe_stop_payload(config, f"missing Step37 inputs: {missing}", env_guard)
        _write_outputs(config, payload)
        return payload

    settings = build_step37_training_settings(config)
    required_ids = sample_ids_by_shard_from_splits(config)
    samples_payload = load_step37_samples(config, required_ids)
    samples_by_shard = samples_payload["samples_by_shard"]
    leakage = summarize_step36_leakage(settings)
    device = _resolve_device(
        str(config["training"].get("device", "cpu")),
        allow_cpu_fallback=bool(config["training"].get("allow_cpu_fallback", True)),
    )
    step36_refs = _step36_reference_by_eval_type(config)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    optimizer_step_performed = False
    for setting in settings:
        row, curve = _train_one_setting(
            config,
            setting,
            samples_by_shard,
            device,
            loss_variant=str(config["loss_ablation"].get("default_variant", "mse_plus_rank_plus_topk_soft")),
            use_proxy_temporal_prior=True,
            optimizer_scope=OPTIMIZER_SCOPE,
            step36_reference_mse=step36_refs.get(setting["eval_type"]),
        )
        rows.append(row)
        curves.append(curve)
        optimizer_step_performed = optimizer_step_performed or bool(row.get("optimizer_step_performed", False))

    loss_ablation = _run_loss_ablation(config, settings, samples_by_shard, device, step36_refs)
    within = aggregate_factorized_metrics([row for row in rows if row["eval_type"] == "within_shard"], eval_type="within_shard")
    cross = aggregate_factorized_metrics([row for row in rows if row["eval_type"] == "cross_shard"], eval_type="cross_shard")
    mixed = aggregate_factorized_metrics([row for row in rows if row["eval_type"] == "mixed_shard"], eval_type="mixed_shard")
    step36_gate = _read_json(config["input"]["step36_gate_decision_json"])
    dataset_bias_detected = bool(step36_gate.get("dataset_bias_detected", False))
    full_context_noise_ack = bool(step36_gate.get("full_context_noise_acknowledged", True))
    top256_min = float(config["future_gates"]["gate_requirements"].get("top256_overlap_mean_min", 0.12))
    gate = build_step37_gate_decision(
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
        "factorized_selector_head_training_performed": True,
        "factorized_selector_with_proxy_temporal_prior_training_performed": True,
        "optimizer_step_performed": bool(optimizer_step_performed),
        "optimizer_scope": OPTIMIZER_SCOPE,
        "loss_ablation_performed": bool(loss_ablation.get("performed", False)),
        "loss_ablation_optimizer_scope": ABLATION_OPTIMIZER_SCOPE,
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
        "factorized_selector_beats_random_token_baseline": bool(gate["factorized_selector_beats_random_token_baseline"]),
        "factorized_selector_beats_uniform_token_baseline": bool(gate["factorized_selector_beats_uniform_token_baseline"]),
        "factorized_selector_beats_temporal_broadcast_baseline": bool(
            gate["factorized_selector_beats_temporal_broadcast_baseline"]
        ),
        "factorized_selector_beats_current_only_patch_baseline": bool(
            gate["factorized_selector_beats_current_only_patch_baseline"]
        ),
        "factorized_selector_beats_step36_direct_patch_selector": bool(
            gate["factorized_selector_beats_step36_direct_patch_selector"]
        ),
        "factorized_selector_generalizes_cross_shard": bool(gate["factorized_selector_generalizes_cross_shard"]),
        "top256_overlap_mean": float(gate["top256_overlap_mean"]),
        "step36_reference_mse_by_eval_type": step36_refs,
        "leakage_checks": leakage,
        "dataset_bias_detected": bool(dataset_bias_detected),
        "full_context_noise_acknowledged": bool(full_context_noise_ack),
        "safety_gate_pass": True,
    }
    payload = {
        "train_summary": train_summary,
        "loss_curves": {"stage": STAGE, "curves": curves, "num_curves": len(curves)},
        "factorized_selector_metrics": metrics,
        "loss_ablation": loss_ablation,
        "gate_decision": gate,
    }
    _write_outputs(config, payload)
    return payload


def _train_one_setting(
    config: dict[str, Any],
    setting: dict[str, Any],
    samples_by_shard: dict[str, list[dict[str, Any]]],
    device: torch.device,
    *,
    loss_variant: str,
    use_proxy_temporal_prior: bool,
    optimizer_scope: str,
    step36_reference_mse: float | None,
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

    _seed_everything(int(config.get("seed", 42)) + int(setting["split_seed"]) + _variant_seed(loss_variant))
    model = _build_model(config, use_proxy_temporal_prior=use_proxy_temporal_prior).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 0.001)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scope = _optimizer_scope(model, optimizer, optimizer_scope)
    if scope["optimizer_scope"] != optimizer_scope:
        raise RuntimeError(f"optimizer scope must be {optimizer_scope}: {scope}")

    train_steps = int(config["training"].get("train_steps", 300))
    eval_every = int(config["training"].get("eval_every", 25))
    batch_size = int(config["training"].get("batch_size", 4))
    eval_batch_size = int(config["training"].get("eval_batch_size", batch_size))
    grad_clip = float(config["training"].get("grad_clip_norm", 1.0))
    topk_values = [int(k) for k in config["metrics"].get("topk_values", [64, 128, 256, 512])]
    aux_weight = float(config["factorized_selector"].get("proxy_temporal_aux_weight", 0.1)) if use_proxy_temporal_prior else 0.0
    rng = random.Random(int(config.get("seed", 42)) + int(setting["split_seed"]) + _variant_seed(loss_variant))
    curve: list[dict[str, Any]] = []
    optimizer_step_performed = False
    for step in range(1, train_steps + 1):
        batch_samples = [train_samples[rng.randrange(len(train_samples))] for _ in range(min(batch_size, len(train_samples)))]
        batch = batch_step37_samples(batch_samples, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["context_tokens"], batch["current_tokens"])
        loss_parts = factorized_selector_loss(
            output=output,
            proxy_patch_target=batch["proxy_patch_target"],
            proxy_temporal_target=batch["proxy_temporal_target"],
            variant=loss_variant,
            config=config["loss_ablation"],
            proxy_temporal_aux_weight=aux_weight,
        )
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
                step36_reference_mse=step36_reference_mse,
            )
            val_eval = _evaluate_samples(
                model,
                val_samples,
                device,
                random_seed=setting["split_seed"] + step + 1000,
                topk_values=topk_values,
                batch_size=eval_batch_size,
                step36_reference_mse=step36_reference_mse,
            )
            curve.append(
                {
                    "step": step,
                    "train_mse": train_eval["factorized_selector_val_mse"],
                    "val_mse": val_eval["factorized_selector_val_mse"],
                    "mse_loss": float(loss_parts["mse_loss"].item()),
                    "rank_loss": float(loss_parts["rank_loss"].item()),
                    "topk_soft_loss": float(loss_parts["topk_soft_loss"].item()),
                    "temporal_aux_loss": float(loss_parts["temporal_aux_loss"].item()),
                }
            )

    final_train = _evaluate_samples(
        model,
        train_samples,
        device,
        random_seed=setting["split_seed"] + 3000,
        topk_values=topk_values,
        batch_size=eval_batch_size,
        step36_reference_mse=step36_reference_mse,
    )
    final_val = _evaluate_samples(
        model,
        val_samples,
        device,
        random_seed=setting["split_seed"] + 4000,
        topk_values=topk_values,
        batch_size=eval_batch_size,
        step36_reference_mse=step36_reference_mse,
    )
    row = {
        "setting": setting["name"],
        "eval_type": setting["eval_type"],
        "split_seed": setting["split_seed"],
        "loss_variant": loss_variant,
        "selector_variant": "factorized_temporal_spatial_with_proxy_temporal_prior"
        if use_proxy_temporal_prior
        else "factorized_temporal_spatial_selector",
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "val_shards": sorted(setting["val_sample_ids_by_shard"].keys()),
        "factorized_selector_val_mse": final_val["factorized_selector_val_mse"],
        "random_token_baseline_mse": final_val["random_token_baseline_mse"],
        "uniform_token_baseline_mse": final_val["uniform_token_baseline_mse"],
        "temporal_broadcast_baseline_mse": final_val["temporal_broadcast_baseline_mse"],
        "current_only_patch_baseline_mse": final_val["current_only_patch_baseline_mse"],
        "step36_direct_patch_selector_mse_if_available": final_val[
            "step36_direct_patch_selector_mse_if_available"
        ],
        "factorized_beats_random": final_val["factorized_beats_random"],
        "factorized_beats_uniform": final_val["factorized_beats_uniform"],
        "factorized_beats_temporal_broadcast": final_val["factorized_beats_temporal_broadcast"],
        "factorized_beats_current_only": final_val["factorized_beats_current_only"],
        "factorized_beats_step36_direct_patch": final_val["factorized_beats_step36_direct_patch"],
        "pearson": final_val["pearson"],
        "spearman": final_val["spearman"],
        "top64_overlap": final_val["top64_overlap"],
        "top128_overlap": final_val["top128_overlap"],
        "top256_overlap": final_val["top256_overlap"],
        "top512_overlap": final_val["top512_overlap"],
        "top256_precision": final_val["top256_precision"],
        "top256_recall": final_val["top256_recall"],
        "val_mse": final_val["val_mse"],
        "val_pearson": final_val["val_pearson"],
        "val_spearman": final_val["val_spearman"],
        "train_mse": final_train["factorized_selector_val_mse"],
        "overfit_gap": float(final_val["factorized_selector_val_mse"] - final_train["factorized_selector_val_mse"]),
        "optimizer_step_performed": bool(optimizer_step_performed),
        "optimizer_scope": optimizer_scope,
        "optimizer_param_count": scope["optimizer_param_count"],
        "videomae_training_performed": False,
        "current_importance_training_performed": False,
        "final_selector_training_performed": False,
        "factorized_selector_head_training_performed": True,
        "checkpoint_saved": False,
        "safety_gate_pass": True,
    }
    return row, {
        "setting": setting["name"],
        "eval_type": setting["eval_type"],
        "split_seed": setting["split_seed"],
        "loss_variant": loss_variant,
        "selector_variant": row["selector_variant"],
        "curve": curve,
    }


def _run_loss_ablation(
    config: dict[str, Any],
    settings: list[dict[str, Any]],
    samples_by_shard: dict[str, list[dict[str, Any]]],
    device: torch.device,
    step36_refs: dict[str, float],
) -> dict[str, Any]:
    if not bool(config["loss_ablation"].get("enabled", True)):
        return {"stage": STAGE, "performed": False, "rows": [], "num_rows": 0}
    variants = [str(variant) for variant in config["loss_ablation"].get("variants", LOSS_VARIANTS)]
    limit = max(1, int(config["selector_variants"].get("loss_ablation_setting_limit", 1)))
    selected = settings[:limit]
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for setting in selected:
        for variant in variants:
            row, curve = _train_one_setting(
                config,
                setting,
                samples_by_shard,
                device,
                loss_variant=variant,
                use_proxy_temporal_prior=False,
                optimizer_scope=ABLATION_OPTIMIZER_SCOPE,
                step36_reference_mse=step36_refs.get(setting["eval_type"]),
            )
            rows.append(row)
            curves.append(curve)
    completed = [row for row in rows if not row.get("skipped")]
    best = min(completed, key=lambda row: float(row["factorized_selector_val_mse"])) if completed else None
    return {
        "stage": STAGE,
        "performed": True,
        "num_rows": len(rows),
        "rows": rows,
        "curves": curves,
        "best_loss_variant": best.get("loss_variant") if best else None,
        "best_factorized_selector_val_mse": best.get("factorized_selector_val_mse") if best else None,
        "optimizer_scope": ABLATION_OPTIMIZER_SCOPE,
        "checkpoint_saved": False,
        "safety_gate_pass": True,
    }


def _evaluate_samples(
    model: ProxyFactorizedSelectorHead,
    samples: list[dict[str, Any]],
    device: torch.device,
    *,
    random_seed: int,
    topk_values: list[int],
    batch_size: int,
    step36_reference_mse: float | None,
) -> dict[str, Any]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk in _chunks(samples, max(1, int(batch_size))):
            batch = batch_step37_samples(chunk, device)
            output = model(batch["context_tokens"], batch["current_tokens"])
            preds.append(torch.sigmoid(output["scores"]).detach().cpu())
            targets.append(batch["proxy_patch_target"].detach().cpu())
    return compute_factorized_prediction_metrics(
        torch.cat(preds, dim=0),
        torch.cat(targets, dim=0),
        random_seed=random_seed,
        topk_values=topk_values,
        step36_reference_mse=step36_reference_mse,
    )


def _build_model(config: dict[str, Any], *, use_proxy_temporal_prior: bool) -> ProxyFactorizedSelectorHead:
    selector_cfg = config["factorized_selector"]
    cls = ProxyFactorizedSelectorWithProxyTemporalPrior if use_proxy_temporal_prior else ProxyFactorizedSelectorHead
    return cls(
        token_dim=int(selector_cfg.get("token_dim", 768)),
        hidden_dim=int(selector_cfg.get("hidden_dim", 128)),
        context_frames=int(selector_cfg.get("context_frames", 16)),
        spatial_tokens=int(selector_cfg.get("spatial_tokens", 392)),
        condition_on_current_summary=bool(selector_cfg.get("condition_on_current_summary", True)),
        use_temporal_embedding=bool(selector_cfg.get("use_temporal_embedding", True)),
        use_spatial_embedding=bool(selector_cfg.get("use_spatial_embedding", True)),
        use_temporal_branch=bool(selector_cfg.get("use_temporal_branch", True)),
        use_spatial_residual_branch=bool(selector_cfg.get("use_spatial_residual_branch", True)),
        combine_mode=str(selector_cfg.get("combine_mode", "temporal_plus_spatial_residual")),
        detach_token_inputs=bool(selector_cfg.get("detach_token_inputs", True)),
    )


def _step36_reference_by_eval_type(config: dict[str, Any]) -> dict[str, float]:
    metrics = _read_json(config["input"]["step36_patch_selector_metrics_json"])
    refs: dict[str, float] = {}
    for key in ("within_shard", "cross_shard", "mixed_shard"):
        payload = metrics.get(key, {})
        value = payload.get("patch_selector_val_mse")
        if value is not None:
            refs[key] = float(value)
    return refs


def _samples_for(ids_by_shard: dict[str, list[str]], samples_by_shard: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for shard, sample_ids in ids_by_shard.items():
        by_id = {str(sample["sample_id"]): sample for sample in samples_by_shard.get(str(shard), [])}
        result.extend(by_id[str(sample_id)] for sample_id in sample_ids if str(sample_id) in by_id)
    return result


def _optimizer_scope(model: torch.nn.Module, optimizer: torch.optim.Optimizer, expected_scope: str) -> dict[str, Any]:
    model_param_ids = {id(param) for param in model.parameters() if param.requires_grad}
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group.get("params", [])
        if getattr(param, "requires_grad", False)
    }
    return {
        "optimizer_scope": expected_scope if optimizer_param_ids == model_param_ids and optimizer_param_ids else "invalid",
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
        "factorized_selector_head_training_performed": False,
        "factorized_selector_with_proxy_temporal_prior_training_performed": False,
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
        "factorized_selector_beats_random_token_baseline": False,
        "factorized_selector_beats_uniform_token_baseline": False,
        "factorized_selector_beats_temporal_broadcast_baseline": False,
        "factorized_selector_generalizes_cross_shard": False,
        "top256_overlap_mean": 0.0,
        "safety_gate_pass": summary["safety_gate_pass"],
    }
    gate = {
        "stage": STAGE,
        "future_factorized_selector_gate_ready": False,
        "factorized_selector_beats_random_token_baseline": False,
        "factorized_selector_beats_uniform_token_baseline": False,
        "factorized_selector_beats_temporal_broadcast_baseline": False,
        "factorized_selector_beats_current_only_patch_baseline": False,
        "factorized_selector_beats_step36_direct_patch_selector": False,
        "factorized_selector_generalizes_cross_shard": False,
        "top256_overlap_mean": 0.0,
        "selector_training_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "reason": reason,
        "recommended_step38": {
            "name": "restore Step33B/Step35/Step36 inputs before Step37",
            "scope": "no downloads or final selector training",
        },
        "safety_gate_pass": summary["safety_gate_pass"],
    }
    return {
        "train_summary": summary,
        "loss_curves": {"stage": STAGE, "curves": [], "num_curves": 0},
        "factorized_selector_metrics": empty,
        "loss_ablation": {"stage": STAGE, "performed": False, "rows": [], "num_rows": 0},
        "gate_decision": gate,
    }


def _write_outputs(config: dict[str, Any], payload: dict[str, Any]) -> None:
    _write_json(config["output"]["train_summary_json"], payload["train_summary"])
    _write_json(config["output"]["loss_curves_json"], payload["loss_curves"])
    _write_json(config["output"]["metrics_json"], payload["factorized_selector_metrics"])
    _write_json(config["output"]["loss_ablation_json"], payload["loss_ablation"])
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


def _variant_seed(name: str) -> int:
    return sum(ord(char) for char in str(name)) % 10_000


def _chunks(samples: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [samples[index : index + size] for index in range(0, len(samples), size)]
