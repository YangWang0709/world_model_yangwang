"""Tiny teacher training utilities for Step 4 sanity checks."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.token_shard_dataset import TokenShardDataset, token_shard_collate_fn
from data.token_shards import load_token_shard
from models.teacher_world_model import TeacherWorldModel
from training.losses import future_latent_mse


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def target_from_future_tokens(future_tokens: torch.Tensor) -> torch.Tensor:
    """Convert future tokens into a `[B, D]` teacher target."""

    if future_tokens.ndim == 3:
        return future_tokens.mean(dim=1)
    if future_tokens.ndim == 2:
        return future_tokens
    raise ValueError(f"future_tokens must be [B, N, D] or [B, D], got {tuple(future_tokens.shape)}")


def grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().data.norm(2).item()
        total += value * value
    return math.sqrt(total)


def save_checkpoint(
    path: str | Path,
    model: TeacherWorldModel,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    model_config: dict[str, Any],
    metrics_summary: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_config": dict(model_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics_summary": metrics_summary or {},
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    required = {"step", "model_config", "model_state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"checkpoint missing required keys: {sorted(missing)}")
    return checkpoint


class TeacherTrainer:
    """Small trainer for full-token teacher sanity checks."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed = int(config.get("seed", 42))
        set_seed(self.seed)

        train_cfg = config["training"]
        output_cfg = config["output"]
        model_cfg = config["model"]
        data_cfg = config["data"]

        self.device = resolve_device(str(train_cfg.get("device", "cuda_if_available")))
        self.max_steps = int(train_cfg.get("max_steps", 30))
        self.log_every = int(train_cfg.get("log_every", 1))
        self.checkpoint_every = int(train_cfg.get("checkpoint_every", self.max_steps))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
        self.run_dir = Path(output_cfg["run_root"]) / output_cfg["run_name"]
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.save_checkpoints = bool(output_cfg.get("save_checkpoint", True))
        self.save_metrics = bool(output_cfg.get("save_metrics", True))

        self.dataset = TokenShardDataset(
            data_cfg["token_shard_dir"],
            shard_glob=data_cfg.get("shard_glob", "tokens_shard_*.pt"),
            map_location="cpu",
        )
        self.dataset_summary = summarize_token_dataset(
            self.dataset,
            split=str(data_cfg.get("split", "")) or None,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=int(train_cfg.get("batch_size", 4)),
            shuffle=True,
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=token_shard_collate_fn,
            drop_last=False,
        )
        self.model_config = {
            "token_dim": int(model_cfg.get("token_dim", 768)),
            "hidden_dim": int(model_cfg.get("hidden_dim", 512)),
            "output_dim": int(model_cfg.get("output_dim", 768)),
            "num_layers": int(model_cfg.get("num_layers", 2)),
            "dropout": float(model_cfg.get("dropout", 0.0)),
            "pool": str(model_cfg.get("pool", "mean")),
        }
        self.model = TeacherWorldModel(**self.model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )

    def train(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.save_metrics:
            self.metrics_path.write_text("", encoding="utf-8")

        metrics: list[dict[str, Any]] = []
        step = 0
        self.model.train()
        while step < self.max_steps:
            for batch in self.loader:
                step += 1
                past_tokens = batch["past_tokens"].to(self.device)
                future_tokens = batch["future_tokens"].to(self.device)
                target = target_from_future_tokens(future_tokens)
                pred = self.model(past_tokens)
                if pred.shape != target.shape:
                    raise ValueError(f"Prediction shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")

                loss = future_latent_mse(pred, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm_before_clip = grad_norm(self.model.parameters())
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metric = {
                    "step": step,
                    "loss": float(loss.item()),
                    "grad_norm": float(norm_before_clip),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                }
                metrics.append(metric)
                if self.save_metrics:
                    with self.metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metric) + "\n")

                if self.log_every > 0 and step % self.log_every == 0:
                    print(f"step={step} loss={metric['loss']:.8f} grad_norm={metric['grad_norm']:.6f}")

                if step >= self.max_steps:
                    break

        losses = [metric["loss"] for metric in metrics]
        summary = {
            "num_steps": len(metrics),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "loss_decreased": bool(losses[-1] < losses[0]),
            "loss_decreased_or_warn": bool(losses[-1] < losses[0]),
            "checkpoint_path": "",
            "metrics_path": str(self.metrics_path),
            "summary_path": str(self.summary_path),
            "device": str(self.device),
            "dataset_size": len(self.dataset),
            **self.dataset_summary,
            "run_dir": str(self.run_dir),
        }

        checkpoint_path = self.checkpoint_dir / f"teacher_world_model_step_{len(metrics):06d}.pt"
        if self.save_checkpoints:
            summary["checkpoint_path"] = str(checkpoint_path)
            save_checkpoint(
                checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                step=len(metrics),
                model_config=self.model_config,
                metrics_summary=summary,
            )

        if bool(self.config["output"].get("save_metrics", True)):
            self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(json.dumps(summary, indent=2))
        return summary


def run_teacher_training(config: dict[str, Any]) -> dict[str, Any]:
    return TeacherTrainer(config).train()


def summarize_token_dataset(dataset: TokenShardDataset, split: str | None = None) -> dict[str, Any]:
    """Return compact source-token metadata for trainer/eval reports."""

    first_sample = dataset[0]
    past_tokens = first_sample["past_tokens"]
    future_tokens = first_sample["future_tokens"]
    if past_tokens.ndim != 2:
        raise ValueError(f"Expected sample past_tokens [N, D], got {tuple(past_tokens.shape)}")
    if future_tokens.ndim not in (1, 2):
        raise ValueError(f"Expected sample future_tokens [D] or [N, D], got {tuple(future_tokens.shape)}")

    first_shard = load_token_shard(dataset.shard_paths[0], map_location="cpu")
    encoder_config = dict(first_shard.get("encoder_config", {}))
    source_encoder = str(first_shard.get("encoder_name", encoder_config.get("actual_encoder", "unknown")))
    source_split = str(split or first_shard.get("split", ""))
    return {
        "num_samples": len(dataset),
        "num_shards": len(dataset.shard_paths),
        "num_tokens": int(past_tokens.shape[0]),
        "token_dim": int(past_tokens.shape[1]),
        "past_token_shape": list(past_tokens.shape),
        "future_token_shape": list(future_tokens.shape),
        "source_encoder": source_encoder,
        "source_split": source_split,
        "token_shard_dir": str(Path(dataset.shard_paths[0]).parent),
        "first_shard_path": str(dataset.shard_paths[0]),
    }
