"""Train UnifiedPredictiveImportanceSelector in context mode only."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.context_token_shard_dataset import ContextSelectorDataset, context_selector_collate_fn, summarize_context_dataset
from models.context_selection_policies import topk_from_retention
from models.unified_predictive_importance_selector import (
    UnifiedPredictiveImportanceSelector,
    build_unified_selector_from_config,
    initialize_context_selector_from_state_checkpoint,
    initialize_context_selector_from_state_checkpoint_with_report,
)
from training.losses import importance_topk_metrics
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


def weighted_mse_alpha(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred/target shapes must match")
    return ((pred - target).pow(2) * (1.0 + float(alpha) * target.float())).mean()


def save_unified_selector_checkpoint(path: str | Path, model: UnifiedPredictiveImportanceSelector, optimizer: torch.optim.Optimizer | None, step: int, model_config: dict[str, Any], summary: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model_config": dict(model_config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics_summary": summary,
            "trained_mode": "context",
            "trained_current_importance": False,
            "init_report": summary.get("init_report", {}),
        },
        out,
    )
    return out


def load_unified_selector_checkpoint(path: str | Path, device: torch.device) -> tuple[UnifiedPredictiveImportanceSelector, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    model = build_unified_selector_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _dataset(config: dict[str, Any], split: str) -> ContextSelectorDataset:
    data_cfg = config["data"]
    max_samples = data_cfg.get(f"max_{split}_samples")
    return ContextSelectorDataset(
        token_shard_dir=data_cfg[f"{split}_token_shard_dir"],
        importance_shard_dir=data_cfg[f"{split}_importance_shard_dir"],
        token_shard_glob=data_cfg.get("token_shard_glob", "context_tokens_shard_*.pt"),
        importance_shard_glob=data_cfg.get("importance_shard_glob", "importance_shard_*.pt"),
        max_samples=int(max_samples) if max_samples is not None else None,
    )


def evaluate_unified_context_selector(model: UnifiedPredictiveImportanceSelector, loader: DataLoader, device: torch.device, topk: int) -> dict[str, float]:
    model.eval()
    scores = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            logits = model(context_tokens=batch["context_tokens"].to(device), current_tokens=batch["current_tokens"].to(device), mode="context")
            scores.append(torch.sigmoid(logits).cpu())
            targets.append(batch["importance_scores_norm"].float().cpu())
    score = torch.cat(scores, dim=0)
    target = torch.cat(targets, dim=0)
    return importance_topk_metrics(score, target, k=topk)


class UnifiedContextSelectorTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        set_seed(int(config.get("seed", 42)))
        self.device = resolve_device(str(config["training"].get("device", "cuda_if_available")))
        self.run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.train_dataset = _dataset(config, "train")
        self.test_dataset = _dataset(config, "test")
        self.train_summary = summarize_context_dataset(self.train_dataset, "train")
        self.test_summary = summarize_context_dataset(self.test_dataset, "test")
        train_cfg = config["training"]
        self.topk = topk_from_retention(
            int(self.train_summary["num_context_tokens"]),
            float(train_cfg.get("retention_ratio", 0.04081632653061224)),
            int(train_cfg.get("fallback_topk_if_unknown", 32)),
        )
        self.loader = DataLoader(self.train_dataset, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)), collate_fn=context_selector_collate_fn)
        self.eval_loader = DataLoader(self.test_dataset, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=0, collate_fn=context_selector_collate_fn)
        self.model_config = dict(config["model"])
        self.model = build_unified_selector_from_config(self.model_config).to(self.device)
        init_cfg = config.get("init", {})
        self.initialized_from_state_selector = False
        self.init_report: dict[str, Any] = {
            "initialized_from_state_selector": False,
            "checkpoint_path": str(init_cfg.get("state_selector_checkpoint", "")),
            "loaded_compatible_key_count": 0,
            "total_legacy_key_count": 0,
            "missing_keys": [],
            "unexpected_keys": [],
            "incompatible_keys": [],
            "error": None,
        }
        if bool(init_cfg.get("init_from_state_selector", False)):
            self.init_report = initialize_context_selector_from_state_checkpoint_with_report(
                self.model,
                init_cfg["state_selector_checkpoint"],
                allow_random_init_if_incompatible=bool(init_cfg.get("allow_random_init_if_incompatible", True)),
            )
            self.initialized_from_state_selector = bool(self.init_report["initialized_from_state_selector"])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
        self.max_steps = int(train_cfg["max_steps"])
        self.alpha = float(train_cfg.get("alpha", 2.0))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))

    def train(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text("", encoding="utf-8")
        metrics = []
        step = 0
        while step < self.max_steps:
            self.model.train()
            for batch in self.loader:
                step += 1
                logits = self.model(context_tokens=batch["context_tokens"].to(self.device), current_tokens=batch["current_tokens"].to(self.device), mode="context")
                probs = torch.sigmoid(logits)
                target = batch["importance_scores_norm"].to(self.device)
                loss = weighted_mse_alpha(probs, target, alpha=self.alpha)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite context selector loss at step {step}")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = grad_norm(self.model.parameters())
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                batch_metrics = importance_topk_metrics(probs.detach().cpu(), target.detach().cpu(), k=self.topk)
                row = {"step": step, "loss": float(loss.item()), "weighted_mse_loss": float(loss.item()), "grad_norm": float(norm), **batch_metrics}
                metrics.append(row)
                with self.metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                if step >= self.max_steps:
                    break
        losses = [row["loss"] for row in metrics]
        eval_metrics = evaluate_unified_context_selector(self.model, self.eval_loader, self.device, self.topk)
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
            "initialized_from_step15_selector": bool(self.initialized_from_state_selector),
            "init_report": self.init_report,
            "trained_mode": "context",
            "trained_current_importance": False,
            "train_state_mode": False,
            "loss_type": "weighted_mse_alpha2",
            "topk": int(self.topk),
            "context_retention_ratio": float(self.topk) / float(self.train_summary["num_context_tokens"]),
            "train_dataset_summary": self.train_summary,
            "test_dataset_summary": self.test_summary,
            "importance_mse": eval_metrics["importance_mse"],
            "pearson_corr": eval_metrics["pearson_corr_mean"],
            "target_topK_overlap": eval_metrics["target_topk_overlap"],
            "selected_context_importance_mean": eval_metrics["selected_teacher_importance_mean"],
            "eval_metrics": eval_metrics,
        }
        checkpoint_path = self.checkpoint_dir / f"unified_selector_context_step_{len(metrics):06d}.pt"
        summary["checkpoint_path"] = str(save_unified_selector_checkpoint(checkpoint_path, self.model, self.optimizer, len(metrics), self.model_config, summary))
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.run_dir / "eval_summary.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary


def run_unified_context_selector_training(config: dict[str, Any]) -> dict[str, Any]:
    return UnifiedContextSelectorTrainer(config).train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "train_unified_context_selector_bair_1000_128.yaml"))
    return parser.parse_args()


def main() -> None:
    run_unified_context_selector_training(load_yaml(parse_args().config))


if __name__ == "__main__":
    main()
