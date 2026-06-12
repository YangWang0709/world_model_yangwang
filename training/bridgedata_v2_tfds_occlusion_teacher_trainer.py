"""Training utilities for Step30B small occlusion teacher."""

from __future__ import annotations

import importlib.util
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from data.bridgedata_v2_tfds_world_model_batch import load_bridge_tfds_world_model_samples, validate_world_model_sample
from models.bridgedata_v2_trained_predictor_teacher import (
    CurrentConditionedContextAttentionPredictor,
    future_token_summary,
    optimizer_scope_for_teacher,
)

TEACHER_SCOPE = "small_predictor_teacher_only"


def train_step30b_teachers_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    env_guard = env_guard_isaaclab()
    missing = missing_step30a_inputs(config)
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"] or missing:
        payload = safe_stop_training_payload(
            config,
            "env_isaaclab is polluted with TensorFlow/TFDS" if not missing else f"missing Step30A inputs: {missing}",
            env_guard,
        )
        write_json(Path(config["output"]["teacher_train_summary_json"]), payload["teacher_train_summary"])
        write_json(Path(config["output"]["teacher_loss_curves_json"]), payload["teacher_loss_curves"])
        return payload

    samples = load_step30a_samples(config)
    splits = json.loads(Path(config["input"]["multiseed_splits_json"]).read_text(encoding="utf-8"))
    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    train_cfg = config["teacher_training"]
    seeds = [int(train_cfg.get("split_seed", 42))]
    if bool(train_cfg.get("also_train_extra_seed_teachers_for_stability", False)):
        seeds.extend(int(seed) for seed in train_cfg.get("extra_split_seeds", []))
    seed_set = set(seeds)
    device = resolve_device(str(train_cfg.get("device", "cpu")), bool(train_cfg.get("allow_cpu_fallback", True)))
    teacher_entries: list[dict[str, Any]] = []
    loss_curves: list[dict[str, Any]] = []
    teachers: dict[int, CurrentConditionedContextAttentionPredictor] = {}
    optimizer_step_performed = False
    optimizer_param_count = 0

    for split in splits.get("splits", []):
        split_seed = int(split["split_seed"])
        if split_seed not in seed_set:
            continue
        train_samples = [samples_by_id[item] for item in split["train_sample_ids"] if item in samples_by_id]
        val_samples = [samples_by_id[item] for item in split["val_sample_ids"] if item in samples_by_id]
        if not train_samples or not val_samples:
            payload = safe_stop_training_payload(config, f"split {split_seed} has no loaded samples", env_guard)
            write_json(Path(config["output"]["teacher_train_summary_json"]), payload["teacher_train_summary"])
            write_json(Path(config["output"]["teacher_loss_curves_json"]), payload["teacher_loss_curves"])
            return payload
        model, curve, scope, performed = train_one_teacher(config, train_samples, val_samples, split_seed, device)
        teachers[split_seed] = model.to("cpu").eval()
        optimizer_step_performed = optimizer_step_performed or performed
        optimizer_param_count = max(optimizer_param_count, int(scope["optimizer_param_count"]))
        loss_curves.append({"split_seed": split_seed, "curve": curve})
        teacher_entries.append(summarize_curve(split_seed, train_samples, val_samples, curve, scope))

    all_val_finite = all(entry["val_loss_finite"] for entry in teacher_entries)
    summary = {
        "stage": config["stage"],
        "safe_stop": False,
        "reason": None,
        "trained_predictor_teacher_training_performed": True,
        "teacher_training_scope": TEACHER_SCOPE,
        "teacher_model_type": str(train_cfg.get("model_type")),
        "split_seeds_trained": [entry["split_seed"] for entry in teacher_entries],
        "num_teachers_trained": len(teacher_entries),
        "teacher_seed_summaries": teacher_entries,
        "teacher_val_loss_finite": bool(all_val_finite),
        "teacher_val_loss_reasonable_decrease": any(entry["val_relative_loss_decrease"] > 0.0 for entry in teacher_entries),
        "optimizer_step_performed": bool(optimizer_step_performed),
        "optimizer_step_scope": TEACHER_SCOPE,
        "optimizer_param_count": int(optimizer_param_count),
        "checkpoint_saved": False,
        "save_teacher_checkpoint": False,
        "save_state_dict": False,
        "token_extraction_performed": False,
        "proxy_importance_regeneration_performed": False,
        "new_tfds_shard_downloaded": False,
        "model_download_performed": False,
        "videomae_training_performed": False,
        "context_teacher_large_training_performed": False,
        "selector_training_performed": False,
        "current_importance_training_performed": False,
        "action_conditioned_world_model_training_performed": False,
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": True,
    }
    payload = {
        "config": config,
        "samples": samples,
        "splits": splits,
        "teachers": teachers,
        "teacher_train_summary": summary,
        "teacher_loss_curves": {"stage": config["stage"], "loss_curves": loss_curves},
    }
    write_json(Path(config["output"]["teacher_train_summary_json"]), summary)
    write_json(Path(config["output"]["teacher_loss_curves_json"]), payload["teacher_loss_curves"])
    return payload


def train_one_teacher(
    config: dict[str, Any],
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    split_seed: int,
    device: torch.device,
) -> tuple[CurrentConditionedContextAttentionPredictor, list[dict[str, Any]], dict[str, Any], bool]:
    train_cfg = config["teacher_training"]
    seed_everything(split_seed)
    model = CurrentConditionedContextAttentionPredictor(
        token_dim=int(train_cfg.get("token_dim", 768)),
        hidden_dim=int(train_cfg.get("hidden_dim", 512)),
        seed=split_seed,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 0.001)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scope = optimizer_scope_for_teacher(model, optimizer)
    if scope["optimizer_step_scope"] != TEACHER_SCOPE:
        raise RuntimeError(f"optimizer scope must be {TEACHER_SCOPE}: {scope}")
    train_steps = int(train_cfg.get("train_steps", 600))
    eval_every = int(train_cfg.get("eval_every", 50))
    batch_size = int(train_cfg.get("train_batch_size", 8))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    curve = [curve_point(0, model, train_samples, val_samples, device, batch_size)]
    performed = False
    rng = random.Random(split_seed)
    for step in range(1, train_steps + 1):
        batch = [train_samples[rng.randrange(len(train_samples))] for _ in range(min(batch_size, len(train_samples)))]
        context, current, target = stack_teacher_batch(batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(context, current)
        loss = F.mse_loss(pred, target, reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        performed = True
        if step == 1 or step % max(eval_every, 1) == 0 or step == train_steps:
            curve.append(curve_point(step, model, train_samples, val_samples, device, batch_size))
    return model, curve, scope, performed


def curve_point(
    step: int,
    model: CurrentConditionedContextAttentionPredictor,
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "train_loss": evaluate_teacher_loss(model, train_samples, device, batch_size),
        "val_loss": evaluate_teacher_loss(model, val_samples, device, batch_size),
    }


def evaluate_teacher_loss(
    model: CurrentConditionedContextAttentionPredictor,
    samples: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    losses: list[float] = []
    counts: list[int] = []
    with torch.no_grad():
        for start in range(0, len(samples), max(1, batch_size)):
            batch = samples[start : start + max(1, batch_size)]
            context, current, target = stack_teacher_batch(batch, device)
            pred = model(context, current)
            per_sample = ((pred - target) ** 2).mean(dim=1)
            losses.append(float(per_sample.sum().item()))
            counts.append(int(per_sample.numel()))
    return float(sum(losses) / max(sum(counts), 1))


def stack_teacher_batch(samples: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context = torch.stack([sample["context_tokens"] for sample in samples], dim=0).to(device=device, dtype=torch.float32)
    current = torch.stack([sample["current_tokens"] for sample in samples], dim=0).to(device=device, dtype=torch.float32)
    future = torch.stack([sample["future_tokens"] for sample in samples], dim=0).to(device=device, dtype=torch.float32)
    return context, current, future_token_summary(future)


def summarize_curve(
    split_seed: int,
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    curve: list[dict[str, Any]],
    scope: dict[str, Any],
) -> dict[str, Any]:
    train_losses = [float(item["train_loss"]) for item in curve]
    val_losses = [float(item["val_loss"]) for item in curve]
    train_initial, train_final = train_losses[0], train_losses[-1]
    val_initial, val_final = val_losses[0], val_losses[-1]
    return {
        "split_seed": int(split_seed),
        "num_train_windows": len(train_samples),
        "num_val_windows": len(val_samples),
        "train_initial_loss": train_initial,
        "train_final_loss": train_final,
        "train_best_loss": min(train_losses),
        "val_initial_loss": val_initial,
        "val_final_loss": val_final,
        "val_best_loss": min(val_losses),
        "train_relative_loss_decrease": relative_decrease(train_initial, train_final),
        "val_relative_loss_decrease": relative_decrease(val_initial, val_final),
        "val_loss_finite": all(math.isfinite(value) for value in val_losses),
        "num_curve_points": len(curve),
        **scope,
    }


def load_step30a_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    loader_config = {
        "input": {
            "token_manifest_jsonl": config["input"]["token_manifest_jsonl"],
            "token_summary_json": config["input"]["token_summary_json"],
            "token_smoke_dir": config["input"]["token_smoke_dir"],
            "importance_manifest_jsonl": config["input"]["proxy_importance_manifest_jsonl"],
            "importance_summary_json": config["input"]["proxy_importance_summary_json"],
            "importance_smoke_dir": config["input"]["proxy_importance_dir"],
        },
        "sample_limits": {"max_samples": int(config["data"].get("num_windows_expected", 64))},
    }
    samples = load_bridge_tfds_world_model_samples(loader_config)
    for sample in samples:
        validate_world_model_sample(sample)
    return samples


def missing_step30a_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["input"]["token_manifest_jsonl"],
        config["input"]["token_summary_json"],
        config["input"]["token_smoke_dir"],
        config["input"]["proxy_importance_manifest_jsonl"],
        config["input"]["proxy_importance_summary_json"],
        config["input"]["proxy_importance_dir"],
        config["input"]["multiseed_splits_json"],
        config["input"]["stability_summary_json"],
        config["input"]["context_signal_decision_json"],
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if not missing:
        token_summary = json.loads(Path(config["input"]["token_summary_json"]).read_text(encoding="utf-8"))
        proxy_summary = json.loads(Path(config["input"]["proxy_importance_summary_json"]).read_text(encoding="utf-8"))
        if not bool(token_summary.get("limited_token_extraction_performed")):
            missing.append("Step30A token_summary reports limited_token_extraction_performed=false")
        if not bool(proxy_summary.get("limited_proxy_importance_generation_performed")):
            missing.append("Step30A proxy importance summary reports limited_proxy_importance_generation_performed=false")
    return missing


def safe_stop_training_payload(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    summary = {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "trained_predictor_teacher_training_performed": False,
        "teacher_training_scope": TEACHER_SCOPE,
        "split_seeds_trained": [],
        "num_teachers_trained": 0,
        "teacher_seed_summaries": [],
        "teacher_val_loss_finite": False,
        "optimizer_step_performed": False,
        "optimizer_step_scope": TEACHER_SCOPE,
        "checkpoint_saved": False,
        "token_extraction_performed": False,
        "proxy_importance_regeneration_performed": False,
        "new_tfds_shard_downloaded": False,
        "model_download_performed": False,
        "videomae_training_performed": False,
        "context_teacher_large_training_performed": False,
        "selector_training_performed": False,
        "current_importance_training_performed": False,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }
    return {
        "config": config,
        "samples": [],
        "splits": {},
        "teachers": {},
        "teacher_train_summary": summary,
        "teacher_loss_curves": {"stage": config["stage"], "loss_curves": []},
    }


def env_guard_isaaclab() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def resolve_device(requested: str, allow_cpu_fallback: bool) -> torch.device:
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


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def relative_decrease(initial: float, final: float) -> float:
    if not math.isfinite(initial) or abs(initial) <= 0.0:
        return 0.0
    return float((initial - final) / abs(initial))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data
