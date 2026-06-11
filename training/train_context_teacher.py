"""Train the Step17 ContextTeacherWorldModel."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.context_token_shard_dataset import ContextTokenShardDataset, context_token_collate_fn, summarize_context_dataset
from models.context_bottleneck_world_model import ContextTeacherWorldModel, future_target_from_tokens
from training.losses import future_latent_mse
from training.teacher_trainer import grad_norm, resolve_device


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_context_teacher_checkpoint(path: str | Path, model: ContextTeacherWorldModel, optimizer: torch.optim.Optimizer | None, step: int, model_config: dict[str, Any], metrics_summary: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model_config": dict(model_config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics_summary": metrics_summary,
        },
        out,
    )
    return out


def load_context_teacher_checkpoint(path: str | Path, device: torch.device) -> tuple[ContextTeacherWorldModel, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    model = ContextTeacherWorldModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _dataset(config: dict[str, Any], split: str) -> ContextTokenShardDataset:
    data_cfg = config["data"]
    root = data_cfg[f"{split}_token_shard_dir"]
    max_samples = data_cfg.get(f"max_{split}_samples")
    return ContextTokenShardDataset(
        root,
        shard_glob=data_cfg.get("shard_glob", "context_tokens_shard_*.pt"),
        max_samples=int(max_samples) if max_samples is not None else None,
    )


def evaluate_context_teacher(model: ContextTeacherWorldModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    sse = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            context = batch["context_tokens"].to(device)
            current = batch["current_tokens"].to(device)
            target = future_target_from_tokens(batch["future_tokens"].to(device))
            pred = model(context, current)
            sse += float((pred - target).pow(2).sum().item())
            count += int(target.numel())
    mse = sse / float(max(count, 1))
    return {"eval_mse": mse, "context_teacher_mse": mse}


class ContextTeacherTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        set_seed(int(config.get("seed", 42)))
        train_cfg = config["training"]
        self.device = resolve_device(str(train_cfg.get("device", "cuda_if_available")))
        self.max_steps = int(train_cfg["max_steps"])
        self.run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.train_dataset = _dataset(config, "train")
        self.test_dataset = _dataset(config, "test")
        self.train_summary = summarize_context_dataset(self.train_dataset, "train")
        self.test_summary = summarize_context_dataset(self.test_dataset, "test")
        self.loader = DataLoader(self.train_dataset, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)), collate_fn=context_token_collate_fn)
        self.eval_loader = DataLoader(self.test_dataset, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=0, collate_fn=context_token_collate_fn)
        self.model_config = {
            "token_dim": int(config["model"].get("token_dim", 768)),
            "hidden_dim": int(config["model"].get("hidden_dim", 512)),
            "output_dim": int(config["model"].get("output_dim", 768)),
            "num_layers": int(config["model"].get("num_layers", 2)),
            "dropout": float(config["model"].get("dropout", 0.0)),
            "current_pool": str(config["model"].get("current_pool", "mean")),
            "context_pool": str(config["model"].get("context_pool", "mean")),
            "fusion": str(config["model"].get("fusion", "concat_mlp")),
        }
        self.model = ContextTeacherWorldModel(**self.model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))

    def train(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text("", encoding="utf-8")
        metrics: list[dict[str, Any]] = []
        step = 0
        while step < self.max_steps:
            self.model.train()
            for batch in self.loader:
                step += 1
                context = batch["context_tokens"].to(self.device)
                current = batch["current_tokens"].to(self.device)
                target = future_target_from_tokens(batch["future_tokens"].to(self.device))
                pred = self.model(context, current)
                loss = future_latent_mse(pred, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite context teacher loss at step {step}")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = grad_norm(self.model.parameters())
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                row = {"step": step, "loss": float(loss.item()), "grad_norm": float(norm)}
                metrics.append(row)
                with self.metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                if step >= self.max_steps:
                    break
        losses = [row["loss"] for row in metrics]
        eval_summary = evaluate_context_teacher(self.model, self.eval_loader, self.device)
        summary = {
            "num_steps": len(metrics),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "loss_decreased": bool(losses[-1] < losses[0]),
            "checkpoint_path": "",
            "metrics_path": str(self.metrics_path),
            "summary_path": str(self.summary_path),
            "run_dir": str(self.run_dir),
            "device": str(self.device),
            "train_num_samples": len(self.train_dataset),
            "test_num_samples": len(self.test_dataset),
            "train_dataset_summary": self.train_summary,
            "test_dataset_summary": self.test_summary,
            **eval_summary,
        }
        checkpoint_path = self.checkpoint_dir / f"context_teacher_step_{len(metrics):06d}.pt"
        summary["checkpoint_path"] = str(save_context_teacher_checkpoint(checkpoint_path, self.model, self.optimizer, len(metrics), self.model_config, summary))
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.run_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary


def run_context_teacher_training(config: dict[str, Any]) -> dict[str, Any]:
    return ContextTeacherTrainer(config).train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "train_context_teacher_bair_1000_128.yaml"))
    return parser.parse_args()


def main() -> None:
    run_context_teacher_training(load_yaml(parse_args().config))


if __name__ == "__main__":
    main()
