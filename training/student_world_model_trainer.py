"""Train a compressed Student world model behind a frozen Student selector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.student_selector_dataset import StudentSelectorDataset, student_selector_collate_fn
from models.attention_selector import AttentionSelector, select_topk
from models.student_world_model import StudentWorldModel
from models.token_compressor import TokenCompressor
from training.losses import future_latent_mse, topk_coverage_metrics
from training.student_selector_trainer import load_student_selector_checkpoint, set_seed
from training.teacher_trainer import grad_norm, resolve_device, target_from_future_tokens


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


def selected_key_metrics(
    scores: torch.Tensor,
    key_token_mask: torch.Tensor,
    k: int,
) -> dict[str, float]:
    """Measure how much key-token signal survives top-k selection."""

    metrics = topk_coverage_metrics(scores, key_token_mask, k=k)
    topk_indices = torch.topk(scores, k=k, dim=1).indices
    selected_key_mask = torch.gather(key_token_mask.float(), dim=1, index=topk_indices)

    coverages = []
    fractions = []
    for sample_selected_mask, sample_key_mask in zip(selected_key_mask, key_token_mask):
        key_count = int(sample_key_mask.float().sum().item())
        if key_count == 0:
            continue
        selected_key_count = float(sample_selected_mask.sum().item())
        coverages.append(selected_key_count / float(key_count))
        fractions.append(selected_key_count / float(k))

    metrics["selected_key_coverage"] = float(sum(coverages) / len(coverages)) if coverages else 0.0
    metrics["selected_key_fraction"] = float(sum(fractions) / len(fractions)) if fractions else 0.0
    return metrics


def build_student_world_model_bundle(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[AttentionSelector, TokenCompressor, StudentWorldModel, dict[str, Any]]:
    checkpoint = load_student_world_model_checkpoint(checkpoint_path, map_location="cpu")

    selector = AttentionSelector(**checkpoint["selector_config"])
    selector_state = checkpoint.get("selector_state_dict")
    if selector_state is None:
        selector_checkpoint = load_student_selector_checkpoint(checkpoint["selector_checkpoint_path"], map_location="cpu")
        selector_state = selector_checkpoint["model_state_dict"]
    selector.load_state_dict(selector_state)
    selector.to(device)
    selector.eval()

    compressor = TokenCompressor(**checkpoint["compressor_config"])
    compressor.load_state_dict(checkpoint["compressor_state_dict"])
    compressor.to(device)
    compressor.eval()

    student_world_model = StudentWorldModel(**checkpoint["student_world_model_config"])
    student_world_model.load_state_dict(checkpoint["student_world_model_state_dict"])
    student_world_model.to(device)
    student_world_model.eval()
    return selector, compressor, student_world_model, checkpoint


def student_world_model_forward(
    selector: AttentionSelector,
    compressor: TokenCompressor,
    student_world_model: StudentWorldModel,
    past_tokens: torch.Tensor,
    topk: int,
    use_sigmoid_scores: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = selector(past_tokens)
    scores = torch.sigmoid(logits) if use_sigmoid_scores else logits
    selected_tokens, selected_indices = select_topk(past_tokens, scores, k=topk)
    compressed_latents = compressor(selected_tokens)
    pred = student_world_model(compressed_latents)
    return pred, scores, selected_indices, compressed_latents


def evaluate_student_world_model_on_loader(
    selector: AttentionSelector,
    compressor: TokenCompressor,
    student_world_model: StudentWorldModel,
    loader: DataLoader,
    device: torch.device,
    topk: int,
    use_sigmoid_scores: bool = True,
) -> dict[str, float]:
    selector.eval()
    compressor.eval()
    student_world_model.eval()
    squared_error_sum = 0.0
    element_count = 0
    all_scores = []
    all_masks = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            past_tokens = batch["past_tokens"].to(device)
            future_tokens = batch["future_tokens"].to(device)
            target = target_from_future_tokens(future_tokens)
            pred, scores, _, _ = student_world_model_forward(
                selector,
                compressor,
                student_world_model,
                past_tokens,
                topk=topk,
                use_sigmoid_scores=use_sigmoid_scores,
            )
            if pred.shape != target.shape:
                raise ValueError(f"Prediction shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
            squared_error_sum += float((pred - target).pow(2).sum().item())
            element_count += int(target.numel())
            all_scores.append(scores.detach().cpu())
            all_masks.append(batch["key_token_mask"].float().cpu())
            all_preds.append(pred.detach().cpu())
            all_targets.append(target.detach().cpu())

    scores_cpu = torch.cat(all_scores, dim=0)
    masks_cpu = torch.cat(all_masks, dim=0)
    preds_cpu = torch.cat(all_preds, dim=0)
    targets_cpu = torch.cat(all_targets, dim=0)
    metrics = selected_key_metrics(scores_cpu, masks_cpu, k=topk)
    return {
        "student_future_mse": squared_error_sum / float(max(element_count, 1)),
        "prediction_mean": float(preds_cpu.mean().item()),
        "prediction_std": float(preds_cpu.std(unbiased=False).item()) if preds_cpu.numel() > 1 else 0.0,
        "target_mean": float(targets_cpu.mean().item()),
        "target_std": float(targets_cpu.std(unbiased=False).item()) if targets_cpu.numel() > 1 else 0.0,
        **metrics,
    }


def save_student_world_model_checkpoint(
    path: str | Path,
    selector: AttentionSelector,
    compressor: TokenCompressor,
    student_world_model: StudentWorldModel,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    selector_checkpoint_path: str,
    selector_config: dict[str, Any],
    compressor_config: dict[str, Any],
    student_world_model_config: dict[str, Any],
    selector_frozen: bool,
    metrics_summary: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "selector_checkpoint_path": str(selector_checkpoint_path),
            "selector_config": dict(selector_config),
            "selector_state_dict": selector.state_dict(),
            "selector_frozen": bool(selector_frozen),
            "compressor_config": dict(compressor_config),
            "compressor_state_dict": compressor.state_dict(),
            "student_world_model_config": dict(student_world_model_config),
            "student_world_model_state_dict": student_world_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics_summary": metrics_summary or {},
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_student_world_model_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    required = {
        "step",
        "selector_checkpoint_path",
        "selector_config",
        "compressor_config",
        "compressor_state_dict",
        "student_world_model_config",
        "student_world_model_state_dict",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"student world model checkpoint missing required keys: {sorted(missing)}")
    return checkpoint


class StudentWorldModelTrainer:
    """Fit TokenCompressor + StudentWorldModel on structured toy future targets."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed = int(config.get("seed", 42))
        set_seed(self.seed)

        train_cfg = config["training"]
        selector_cfg = config["selector"]
        compressor_cfg = config["compressor"]
        student_cfg = config["student_world_model"]
        output_cfg = config["output"]

        self.device = resolve_device(str(train_cfg.get("device", "cuda_if_available")))
        self.max_steps = int(train_cfg.get("max_steps", 300))
        self.log_every = int(train_cfg.get("log_every", 1))
        self.checkpoint_every = int(train_cfg.get("checkpoint_every", self.max_steps))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
        self.topk = int(train_cfg.get("topk", 4))
        self.use_sigmoid_scores = bool(train_cfg.get("use_sigmoid_scores", True))
        self.selector_frozen = bool(selector_cfg.get("frozen", True))
        self.selector_checkpoint_path = str(selector_cfg["checkpoint"])

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

        selector_checkpoint = load_student_selector_checkpoint(self.selector_checkpoint_path, map_location="cpu")
        self.selector_config = dict(selector_checkpoint["model_config"])
        self.selector = AttentionSelector(**self.selector_config)
        self.selector.load_state_dict(selector_checkpoint["model_state_dict"])
        self.selector.to(self.device)
        for parameter in self.selector.parameters():
            parameter.requires_grad = not self.selector_frozen

        self.compressor_config = {
            "token_dim": int(compressor_cfg.get("token_dim", 768)),
            "latent_dim": int(compressor_cfg.get("latent_dim", 512)),
            "num_latents": int(compressor_cfg.get("num_latents", 16)),
            "hidden_dim": int(compressor_cfg.get("hidden_dim", 512)),
            "dropout": float(compressor_cfg.get("dropout", 0.0)),
            "compressor_type": str(compressor_cfg.get("compressor_type", "perceiver_like")),
            "num_heads": int(compressor_cfg.get("num_heads", 8)),
        }
        self.student_world_model_config = {
            "latent_dim": int(student_cfg.get("latent_dim", self.compressor_config["latent_dim"])),
            "hidden_dim": int(student_cfg.get("hidden_dim", 512)),
            "output_dim": int(student_cfg.get("output_dim", 768)),
            "num_layers": int(student_cfg.get("num_layers", 2)),
            "dropout": float(student_cfg.get("dropout", 0.0)),
            "pool": str(student_cfg.get("pool", "mean")),
        }
        self.compressor = TokenCompressor(**self.compressor_config).to(self.device)
        self.student_world_model = StudentWorldModel(**self.student_world_model_config).to(self.device)

        params = list(self.compressor.parameters()) + list(self.student_world_model.parameters())
        if not self.selector_frozen:
            params += list(self.selector.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(train_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )

    def _set_train_mode(self) -> None:
        self.compressor.train()
        self.student_world_model.train()
        if self.selector_frozen:
            self.selector.eval()
        else:
            self.selector.train()

    def train(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.save_metrics:
            self.metrics_path.write_text("", encoding="utf-8")

        metrics: list[dict[str, Any]] = []
        step = 0
        while step < self.max_steps:
            self._set_train_mode()
            for batch in self.loader:
                step += 1
                past_tokens = batch["past_tokens"].to(self.device)
                future_tokens = batch["future_tokens"].to(self.device)
                key_mask = batch["key_token_mask"].to(self.device)
                target = target_from_future_tokens(future_tokens)

                if self.selector_frozen:
                    with torch.no_grad():
                        logits = self.selector(past_tokens)
                        scores = torch.sigmoid(logits) if self.use_sigmoid_scores else logits
                    selected_tokens, _ = select_topk(past_tokens, scores, k=self.topk)
                    compressed_latents = self.compressor(selected_tokens)
                    pred = self.student_world_model(compressed_latents)
                else:
                    pred, scores, _, compressed_latents = student_world_model_forward(
                        self.selector,
                        self.compressor,
                        self.student_world_model,
                        past_tokens,
                        topk=self.topk,
                        use_sigmoid_scores=self.use_sigmoid_scores,
                    )

                if pred.shape != target.shape:
                    raise ValueError(f"Prediction shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
                loss = future_latent_mse(pred, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite student world model loss at step {step}: {loss.item()}")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm_before_clip = grad_norm(
                    list(self.compressor.parameters())
                    + list(self.student_world_model.parameters())
                    + ([] if self.selector_frozen else list(self.selector.parameters()))
                )
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.compressor.parameters())
                        + list(self.student_world_model.parameters())
                        + ([] if self.selector_frozen else list(self.selector.parameters())),
                        self.max_grad_norm,
                    )
                self.optimizer.step()

                batch_metrics = selected_key_metrics(scores.detach().cpu(), key_mask.detach().cpu(), k=self.topk)
                metric = {
                    "step": int(step),
                    "loss": float(loss.item()),
                    "grad_norm": float(norm_before_clip),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    "compressed_latents": int(compressed_latents.shape[1]),
                    "latent_dim": int(compressed_latents.shape[2]),
                    **batch_metrics,
                }
                metrics.append(metric)
                if self.save_metrics:
                    with self.metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metric) + "\n")
                if self.log_every > 0 and step % self.log_every == 0:
                    print(
                        "step={step} loss={loss:.8f} grad_norm={grad:.6f} "
                        "top1={top1:.3f} topk={topk:.3f} coverage={coverage:.3f}".format(
                            step=step,
                            loss=metric["loss"],
                            grad=metric["grad_norm"],
                            top1=metric["top1_hit_rate"],
                            topk=metric["topk_hit_rate"],
                            coverage=metric["selected_key_coverage"],
                        )
                    )
                if step >= self.max_steps:
                    break

        losses = [metric["loss"] for metric in metrics]
        eval_metrics = evaluate_student_world_model_on_loader(
            self.selector,
            self.compressor,
            self.student_world_model,
            self.eval_loader,
            device=self.device,
            topk=self.topk,
            use_sigmoid_scores=self.use_sigmoid_scores,
        )
        summary = {
            "num_steps": len(metrics),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "loss_decreased": bool(losses[-1] < losses[0]),
            "student_future_mse": eval_metrics["student_future_mse"],
            "final_selected_top1_hit_rate": eval_metrics["top1_hit_rate"],
            "final_selected_topk_hit_rate": eval_metrics["topk_hit_rate"],
            "final_selected_key_coverage": eval_metrics["selected_key_coverage"],
            "final_selected_key_fraction": eval_metrics["selected_key_fraction"],
            "token_retention_ratio": float(self.topk) / float(self.dataset[0]["past_tokens"].shape[0]),
            "checkpoint_path": "",
            "metrics_path": str(self.metrics_path),
            "summary_path": str(self.summary_path),
            "device": str(self.device),
            "dataset_size": len(self.dataset),
            "run_dir": str(self.run_dir),
            "selector_checkpoint_path": self.selector_checkpoint_path,
            "selector_frozen": bool(self.selector_frozen),
            "topk": int(self.topk),
            "compressed_tokens": int(self.compressor_config["num_latents"]),
            "student_world_model_eval": eval_metrics,
        }

        checkpoint_path = self.checkpoint_dir / f"student_world_model_step_{len(metrics):06d}.pt"
        if self.save_checkpoints:
            summary["checkpoint_path"] = str(checkpoint_path)
            save_student_world_model_checkpoint(
                checkpoint_path,
                selector=self.selector,
                compressor=self.compressor,
                student_world_model=self.student_world_model,
                optimizer=self.optimizer,
                step=len(metrics),
                selector_checkpoint_path=self.selector_checkpoint_path,
                selector_config=self.selector_config,
                compressor_config=self.compressor_config,
                student_world_model_config=self.student_world_model_config,
                selector_frozen=self.selector_frozen,
                metrics_summary=summary,
            )

        if self.save_metrics:
            self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary


def run_student_world_model_training(config: dict[str, Any]) -> dict[str, Any]:
    return StudentWorldModelTrainer(config).train()
