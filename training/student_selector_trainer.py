"""Tiny Student attention selector training for structured toy importance labels."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.student_selector_dataset import StudentSelectorDataset, student_selector_collate_fn
from models.attention_selector import AttentionSelector
from training.losses import importance_regression_loss, ranking_margin_loss, topk_coverage_metrics
from training.teacher_trainer import grad_norm, resolve_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_student_selector_checkpoint(
    path: str | Path,
    model: AttentionSelector,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    model_config: dict[str, Any],
    metrics_summary: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model_config": dict(model_config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics_summary": metrics_summary or {},
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_student_selector_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    missing = {"step", "model_config", "model_state_dict"}.difference(checkpoint)
    if missing:
        raise KeyError(f"student selector checkpoint missing required keys: {sorted(missing)}")
    return checkpoint


def _dataset_from_config(config: dict[str, Any]) -> StudentSelectorDataset:
    data_cfg = config["data"]
    return StudentSelectorDataset(
        token_shard_dir=data_cfg["token_shard_dir"],
        importance_shard_dir=data_cfg["importance_shard_dir"],
        token_shard_glob=data_cfg.get("token_shard_glob", "tokens_shard_*.pt"),
        importance_shard_glob=data_cfg.get("importance_shard_glob", "importance_shard_*.pt"),
        require_key_token_mask=bool(data_cfg.get("require_key_token_mask", True)),
        map_location="cpu",
    )


def evaluate_selector_on_loader(
    model: AttentionSelector,
    loader: DataLoader,
    device: torch.device,
    topk: int,
) -> dict[str, float]:
    model.eval()
    all_scores = []
    all_targets = []
    all_masks = []
    with torch.no_grad():
        for batch in loader:
            past_tokens = batch["past_tokens"].to(device)
            logits = model(past_tokens)
            scores = torch.sigmoid(logits).detach().cpu()
            all_scores.append(scores)
            all_targets.append(batch["importance_scores_norm"].float().cpu())
            all_masks.append(batch["key_token_mask"].float().cpu())
    scores = torch.cat(all_scores, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)
    coverage = topk_coverage_metrics(scores, masks, k=topk)
    return {
        "importance_mse": float(torch.nn.functional.mse_loss(scores, targets).item()),
        "score_mean": float(scores.mean().item()),
        "score_std": float(scores.std(unbiased=False).item()) if scores.numel() > 1 else 0.0,
        "score_min": float(scores.min().item()),
        "score_max": float(scores.max().item()),
        "target_mean": float(targets.mean().item()),
        "target_std": float(targets.std(unbiased=False).item()) if targets.numel() > 1 else 0.0,
        "target_min": float(targets.min().item()),
        "target_max": float(targets.max().item()),
        **coverage,
    }


class StudentSelectorTrainer:
    """Small trainer that fits selector scores to Step 5.5 importance labels."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed = int(config.get("seed", 42))
        set_seed(self.seed)

        train_cfg = config["training"]
        model_cfg = config["model"]
        output_cfg = config["output"]

        self.device = resolve_device(str(train_cfg.get("device", "cuda_if_available")))
        self.max_steps = int(train_cfg.get("max_steps", 200))
        self.log_every = int(train_cfg.get("log_every", 1))
        self.checkpoint_every = int(train_cfg.get("checkpoint_every", self.max_steps))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
        self.importance_loss_type = str(train_cfg.get("importance_loss_type", "mse"))
        self.importance_loss_weight = float(train_cfg.get("importance_loss_weight", 1.0))
        self.ranking_loss_weight = float(train_cfg.get("ranking_loss_weight", 0.1))
        self.ranking_margin = float(train_cfg.get("ranking_margin", 0.1))
        self.topk = int(train_cfg.get("topk", 4))

        self.run_dir = Path(output_cfg["run_root"]) / output_cfg["run_name"]
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.save_checkpoints = bool(output_cfg.get("save_checkpoint", True))
        self.save_metrics = bool(output_cfg.get("save_metrics", True))

        self.dataset = _dataset_from_config(config)
        self.loader = DataLoader(
            self.dataset,
            batch_size=int(train_cfg.get("batch_size", 8)),
            shuffle=True,
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=student_selector_collate_fn,
            drop_last=False,
        )
        self.eval_loader = DataLoader(
            self.dataset,
            batch_size=int(train_cfg.get("batch_size", 8)),
            shuffle=False,
            num_workers=0,
            collate_fn=student_selector_collate_fn,
            drop_last=False,
        )
        self.model_config = {
            "token_dim": int(model_cfg.get("token_dim", 768)),
            "hidden_dim": int(model_cfg.get("hidden_dim", 256)),
            "task_dim": model_cfg.get("task_dim"),
            "use_task": bool(model_cfg.get("use_task", False)),
            "dropout": float(model_cfg.get("dropout", 0.0)),
        }
        self.model = AttentionSelector(**self.model_config).to(self.device)
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
                targets = batch["importance_scores_norm"].to(self.device)
                key_mask = batch["key_token_mask"].to(self.device)
                logits = self.model(past_tokens)
                score_probs = torch.sigmoid(logits)
                imp_loss = importance_regression_loss(
                    score_probs,
                    targets,
                    loss_type=self.importance_loss_type,
                )
                rank_loss = ranking_margin_loss(score_probs, key_mask, margin=self.ranking_margin)
                loss = self.importance_loss_weight * imp_loss + self.ranking_loss_weight * rank_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite selector loss at step {step}: {loss.item()}")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm_before_clip = grad_norm(self.model.parameters())
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                coverage = topk_coverage_metrics(score_probs.detach().cpu(), key_mask.detach().cpu(), k=self.topk)
                metric = {
                    "step": int(step),
                    "loss": float(loss.item()),
                    "importance_loss": float(imp_loss.item()),
                    "ranking_loss": float(rank_loss.item()),
                    "grad_norm": float(norm_before_clip),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    **coverage,
                }
                metrics.append(metric)
                if self.save_metrics:
                    with self.metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metric) + "\n")
                if self.log_every > 0 and step % self.log_every == 0:
                    print(
                        "step={step} loss={loss:.8f} imp={imp:.8f} rank={rank:.8f} "
                        "gap={gap:.6f} top1={top1:.3f} topk={topk:.3f}".format(
                            step=step,
                            loss=metric["loss"],
                            imp=metric["importance_loss"],
                            rank=metric["ranking_loss"],
                            gap=metric["key_vs_non_key_gap"],
                            top1=metric["top1_hit_rate"],
                            topk=metric["topk_hit_rate"],
                        )
                    )
                if step >= self.max_steps:
                    break

        losses = [metric["loss"] for metric in metrics]
        eval_metrics = evaluate_selector_on_loader(self.model, self.eval_loader, self.device, self.topk)
        summary = {
            "num_steps": len(metrics),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "loss_decreased": bool(losses[-1] < losses[0]),
            "final_key_score_mean": eval_metrics["key_score_mean"],
            "final_non_key_score_mean": eval_metrics["non_key_score_mean"],
            "final_key_vs_non_key_gap": eval_metrics["key_vs_non_key_gap"],
            "final_top1_hit_rate": eval_metrics["top1_hit_rate"],
            "final_topk_hit_rate": eval_metrics["topk_hit_rate"],
            "final_importance_mse": eval_metrics["importance_mse"],
            "checkpoint_path": "",
            "metrics_path": str(self.metrics_path),
            "summary_path": str(self.summary_path),
            "device": str(self.device),
            "dataset_size": len(self.dataset),
            "run_dir": str(self.run_dir),
            "topk": int(self.topk),
        }

        checkpoint_path = self.checkpoint_dir / f"student_selector_step_{len(metrics):06d}.pt"
        if self.save_checkpoints:
            summary["checkpoint_path"] = str(checkpoint_path)
            save_student_selector_checkpoint(
                checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                step=len(metrics),
                model_config=self.model_config,
                metrics_summary=summary,
            )

        if self.save_metrics:
            self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary


def run_student_selector_training(config: dict[str, Any]) -> dict[str, Any]:
    return StudentSelectorTrainer(config).train()
