"""Train Step 16 downstream utilization variants on fixed BAIR tokens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.student_selector_dataset import student_selector_collate_fn
from models.attention_selector import AttentionSelector
from models.downstream_utilization_modules import BaseDownstreamUtilizer, build_downstream_utilizer
from models.selection_policies import (
    compute_teacher_importance_selection_metrics,
    gather_tokens_by_indices,
)
from models.teacher_world_model import TeacherWorldModel
from training.losses import future_latent_mse
from training.student_selector_trainer import load_student_selector_checkpoint, set_seed
from training.student_world_model_trainer import (
    build_student_world_model_dataset,
    summarize_student_world_model_dataset,
)
from training.teacher_trainer import grad_norm, load_checkpoint, resolve_device, target_from_future_tokens


def _is_finite(value: Any) -> bool:
    try:
        return torch.isfinite(torch.tensor(float(value))).item()
    except (TypeError, ValueError):
        return False


def _importance_from_batch(batch: dict[str, Any]) -> torch.Tensor | None:
    if "importance_scores_norm" in batch:
        return batch["importance_scores_norm"].float()
    if "importance_scores" in batch:
        return batch["importance_scores"].float()
    return None


def selector_checkpoint_path(config: dict[str, Any], seed: int) -> Path:
    selector_cfg = config["learned_selectors"]
    root = Path(selector_cfg["root"])
    pattern = str(selector_cfg.get("checkpoint_pattern", "weighted_mse_alpha2_seed{seed}/checkpoints/student_selector_step_001000.pt"))
    path = root / pattern.format(seed=int(seed))
    if path.exists():
        return path
    fallback = root / f"weighted_mse_alpha2_seed{int(seed)}" / "checkpoints" / "student_selector_step_001000.pt"
    return fallback


def load_frozen_selector(checkpoint_path: str | Path, device: torch.device) -> AttentionSelector:
    checkpoint = load_student_selector_checkpoint(checkpoint_path, map_location="cpu")
    selector = AttentionSelector(**checkpoint["model_config"])
    selector.load_state_dict(checkpoint["model_state_dict"])
    selector.to(device)
    selector.eval()
    for parameter in selector.parameters():
        parameter.requires_grad = False
    return selector


def load_teacher(checkpoint_path: str | Path, device: torch.device) -> TeacherWorldModel:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    teacher = TeacherWorldModel(**checkpoint["model_config"])
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def _uniform_indices(batch_size: int, num_tokens: int, k: int, device: torch.device) -> torch.Tensor:
    base = torch.linspace(0, num_tokens - 1, steps=k).round().long()
    return base.unsqueeze(0).expand(batch_size, -1).to(device)


def _combine_unique_indices(
    learned_sorted: torch.Tensor,
    uniform_indices: torch.Tensor,
    learned_k: int,
    uniform_k: int,
    topk: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for learned_row, uniform_row in zip(learned_sorted.cpu(), uniform_indices.cpu()):
        values: list[int] = []
        for index in learned_row[:learned_k].tolist():
            if int(index) not in values:
                values.append(int(index))
        for index in uniform_row[:uniform_k].tolist():
            if int(index) not in values:
                values.append(int(index))
        for index in learned_row.tolist():
            if len(values) >= topk:
                break
            if int(index) not in values:
                values.append(int(index))
        if len(values) != topk:
            raise ValueError(f"Could not build {topk} unique hybrid indices")
        rows.append(torch.tensor(values, dtype=torch.long))
    return torch.stack(rows, dim=0)


def select_downstream_tokens(
    *,
    selector: AttentionSelector,
    past_tokens: torch.Tensor,
    variant: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor | str]:
    policy = str(variant.get("selection_policy", "learned_top16"))
    topk = int(variant.get("topk", 16))
    if topk <= 0 or topk > past_tokens.shape[1]:
        raise ValueError(f"topk must be in [1, {past_tokens.shape[1]}], got {topk}")
    selector.eval()
    with torch.no_grad():
        logits = selector(past_tokens.to(device))
        scores = torch.sigmoid(logits)
    if policy == "learned_top16":
        selected_indices = torch.topk(scores, k=topk, dim=1).indices
    elif policy == "hybrid_learned8_uniform8":
        learned_k = int(variant.get("learned_k", topk // 2))
        uniform_k = int(variant.get("uniform_k", topk - learned_k))
        learned_sorted = torch.topk(scores, k=min(past_tokens.shape[1], max(topk, learned_k + uniform_k + topk)), dim=1).indices
        uniform = _uniform_indices(past_tokens.shape[0], past_tokens.shape[1], uniform_k, device=scores.device)
        selected_indices = _combine_unique_indices(learned_sorted, uniform, learned_k, uniform_k, topk).to(device)
    else:
        raise ValueError(f"Unsupported Step16 selection_policy: {policy!r}")
    selected_tokens = gather_tokens_by_indices(past_tokens, selected_indices)
    selected_scores = torch.gather(scores, dim=1, index=selected_indices)
    return {
        "selected_tokens": selected_tokens,
        "selected_indices": selected_indices,
        "selected_scores": selected_scores,
        "scores": scores,
        "policy_name": policy,
    }


def utilizer_kwargs_from_config(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    compressor_cfg = config.get("compressor", {})
    student_cfg = config.get("student_world_model", {})
    return {
        "token_dim": int(compressor_cfg.get("token_dim", 768)),
        "latent_dim": int(variant.get("latent_dim", compressor_cfg.get("latent_dim", 512))),
        "hidden_dim": int(student_cfg.get("hidden_dim", compressor_cfg.get("hidden_dim", 512))),
        "output_dim": int(student_cfg.get("output_dim", 768)),
        "num_latents": int(variant.get("num_latents", compressor_cfg.get("num_latents", 16))),
        "dropout": float(student_cfg.get("dropout", compressor_cfg.get("dropout", 0.0))),
        "compressor_type": str(compressor_cfg.get("compressor_type", "perceiver_like")),
        "num_heads": int(variant.get("num_heads", compressor_cfg.get("num_heads", 8))),
        "num_layers": int(student_cfg.get("num_layers", 2)),
        "pool": str(student_cfg.get("pool", "mean")),
        "transformer_layers": int(variant.get("transformer_layers", 2)),
        "transformer_heads": int(variant.get("transformer_heads", variant.get("num_heads", 8))),
        "cross_attention_layers": int(variant.get("cross_attention_layers", 2)),
    }


def build_utilizer_from_variant(config: dict[str, Any], variant: dict[str, Any]) -> BaseDownstreamUtilizer:
    return build_downstream_utilizer(str(variant["utilization_module"]), **utilizer_kwargs_from_config(config, variant))


def save_downstream_checkpoint(
    path: str | Path,
    utilizer: BaseDownstreamUtilizer,
    optimizer: torch.optim.Optimizer | None,
    summary: dict[str, Any],
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(summary["num_steps"]),
            "variant": summary["variant"],
            "phase": summary["phase"],
            "seed": int(summary["seed"]),
            "utilizer_config": summary["utilizer_config"],
            "utilizer_state_dict": utilizer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics_summary": summary,
        },
        checkpoint_path,
    )
    return checkpoint_path


def evaluate_downstream_utilizer(
    *,
    selector: AttentionSelector,
    utilizer: BaseDownstreamUtilizer,
    teacher: TeacherWorldModel,
    loader: DataLoader,
    device: torch.device,
    variant: dict[str, Any],
    random_seed: int,
) -> dict[str, float | None]:
    selector.eval()
    utilizer.eval()
    teacher.eval()
    student_sse = 0.0
    teacher_sse = 0.0
    element_count = 0
    all_indices: list[torch.Tensor] = []
    all_importance: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            past_tokens = batch["past_tokens"].to(device)
            future_tokens = batch["future_tokens"].to(device)
            target = target_from_future_tokens(future_tokens)
            selection = select_downstream_tokens(selector=selector, past_tokens=past_tokens, variant=variant, device=device)
            pred = utilizer(
                selection["selected_tokens"],  # type: ignore[arg-type]
                selected_scores=selection["selected_scores"],  # type: ignore[arg-type]
                metadata={"variant": variant},
            )
            teacher_pred = teacher(past_tokens)
            if pred.shape != target.shape:
                raise ValueError(f"Prediction shape {tuple(pred.shape)} != target {tuple(target.shape)}")
            if teacher_pred.shape != target.shape:
                raise ValueError(f"Teacher prediction shape {tuple(teacher_pred.shape)} != target {tuple(target.shape)}")
            student_sse += float((pred - target).pow(2).sum().item())
            teacher_sse += float((teacher_pred - target).pow(2).sum().item())
            element_count += int(target.numel())
            all_indices.append(selection["selected_indices"].detach().cpu())  # type: ignore[union-attr]
            importance = _importance_from_batch(batch)
            if importance is not None:
                all_importance.append(importance.detach().cpu())
    student_mse = student_sse / float(max(element_count, 1))
    teacher_mse = teacher_sse / float(max(element_count, 1))
    selected_indices = torch.cat(all_indices, dim=0)
    num_tokens = int(variant.get("num_tokens", 392))
    topk = int(variant.get("topk", selected_indices.shape[1]))
    metrics: dict[str, float | None] = {
        "student_future_mse": student_mse,
        "student_mse": student_mse,
        "teacher_mse": teacher_mse,
        "teacher_future_mse": teacher_mse,
        "student_teacher_gap": student_mse - teacher_mse,
        "student_teacher_ratio": student_mse / max(teacher_mse, 1e-12),
        "token_retention_ratio": float(topk) / float(num_tokens),
    }
    if all_importance:
        importance_tensor = torch.cat(all_importance, dim=0)
        metrics.update(
            compute_teacher_importance_selection_metrics(
                selected_indices,
                importance_tensor,
                k=topk,
                num_tokens=int(importance_tensor.shape[1]),
                random_seed=random_seed,
            )
        )
    else:
        metrics.update(
            {
                "selector_target_top1_overlap": None,
                "selector_target_topk_overlap": None,
                "selected_teacher_importance_mean": None,
                "random_teacher_importance_mean": None,
                "selected_vs_random_importance_gap": None,
            }
        )
    return metrics


class DownstreamUtilizationTrainer:
    """Train one Step16 utilization variant with a frozen selector."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        phase: str,
        variant: dict[str, Any],
        seed: int,
        max_steps: int,
        batch_size: int,
        num_workers: int,
        run_dir: str | Path,
    ) -> None:
        self.config = config
        self.phase = phase
        self.variant = dict(variant)
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        set_seed(int(config.get("seed", 42)) + self.seed)

        self.device = resolve_device(str(config.get("training", {}).get("device", "cuda_if_available")))
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.eval_summary_path = self.run_dir / "eval_summary.json"
        self.gap_summary_path = self.run_dir / "teacher_student_gap_summary.json"

        self.train_dataset = build_student_world_model_dataset(config, split="train")
        self.eval_dataset = build_student_world_model_dataset(config, split="test")
        self.train_dataset_summary = summarize_student_world_model_dataset(self.train_dataset, split="train")
        self.eval_dataset_summary = summarize_student_world_model_dataset(self.eval_dataset, split="test")
        generator = torch.Generator()
        generator.manual_seed(int(config.get("seed", 42)) + self.seed)
        self.loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=student_selector_collate_fn,
            drop_last=False,
            generator=generator,
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=student_selector_collate_fn,
            drop_last=False,
        )

        self.selector_checkpoint = selector_checkpoint_path(config, self.seed)
        self.selector = load_frozen_selector(self.selector_checkpoint, self.device)
        self.teacher = load_teacher(config["teacher_reference"]["checkpoint"], self.device)
        self.utilizer_config = utilizer_kwargs_from_config(config, self.variant)
        self.utilizer = build_utilizer_from_variant(config, self.variant).to(self.device)
        student_cfg = config["student_world_model"]
        self.optimizer = torch.optim.AdamW(
            self.utilizer.parameters(),
            lr=float(student_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(student_cfg.get("weight_decay", 1e-4)),
        )
        self.max_grad_norm = float(student_cfg.get("max_grad_norm", 0.0))

    def train(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text("", encoding="utf-8")

        metrics: list[dict[str, Any]] = []
        step = 0
        while step < self.max_steps:
            self.utilizer.train()
            self.selector.eval()
            for batch in self.loader:
                step += 1
                past_tokens = batch["past_tokens"].to(self.device)
                future_tokens = batch["future_tokens"].to(self.device)
                target = target_from_future_tokens(future_tokens)
                selection = select_downstream_tokens(
                    selector=self.selector,
                    past_tokens=past_tokens,
                    variant=self.variant,
                    device=self.device,
                )
                pred = self.utilizer(
                    selection["selected_tokens"],  # type: ignore[arg-type]
                    selected_scores=selection["selected_scores"],  # type: ignore[arg-type]
                    metadata={"variant": self.variant, "phase": self.phase},
                )
                if pred.shape != target.shape:
                    raise ValueError(f"Prediction shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
                loss = future_latent_mse(pred, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite Step16 loss at step {step}: {loss.item()}")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm_before_clip = grad_norm(self.utilizer.parameters())
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.utilizer.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metric = {
                    "step": int(step),
                    "phase": self.phase,
                    "variant": self.variant["name"],
                    "seed": int(self.seed),
                    "loss": float(loss.item()),
                    "grad_norm": float(norm_before_clip),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                }
                metrics.append(metric)
                with self.metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metric) + "\n")
                if step >= self.max_steps:
                    break

        losses = [metric["loss"] for metric in metrics]
        eval_metrics = evaluate_downstream_utilizer(
            selector=self.selector,
            utilizer=self.utilizer,
            teacher=self.teacher,
            loader=self.eval_loader,
            device=self.device,
            variant=self.variant,
            random_seed=self.seed,
        )
        summary = {
            "success": True,
            "phase": self.phase,
            "variant": self.variant["name"],
            "seed": int(self.seed),
            "selection_policy": self.variant.get("selection_policy"),
            "utilization_module": self.variant.get("utilization_module"),
            "topk": int(self.variant.get("topk", self.config["selection"]["topk"])),
            "num_steps": len(metrics),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "loss_decreased": bool(losses[-1] < losses[0]),
            "student_future_mse": eval_metrics["student_future_mse"],
            "teacher_mse": eval_metrics["teacher_mse"],
            "student_teacher_ratio": eval_metrics["student_teacher_ratio"],
            "selector_target_topk_overlap": eval_metrics.get("selector_target_topk_overlap"),
            "selector_target_top1_overlap": eval_metrics.get("selector_target_top1_overlap"),
            "selected_teacher_importance_mean": eval_metrics.get("selected_teacher_importance_mean"),
            "random_teacher_importance_mean": eval_metrics.get("random_teacher_importance_mean"),
            "selected_vs_random_importance_gap": eval_metrics.get("selected_vs_random_importance_gap"),
            "token_retention_ratio": eval_metrics.get("token_retention_ratio"),
            "checkpoint_path": "",
            "metrics_path": str(self.metrics_path),
            "summary_path": str(self.summary_path),
            "eval_summary_path": str(self.eval_summary_path),
            "gap_summary_path": str(self.gap_summary_path),
            "run_dir": str(self.run_dir),
            "device": str(self.device),
            "selector_checkpoint": str(self.selector_checkpoint),
            "teacher_checkpoint_path": str(self.config["teacher_reference"]["checkpoint"]),
            "train_num_samples": len(self.train_dataset),
            "test_num_samples": len(self.eval_dataset),
            "num_tokens": int(self.train_dataset[0]["past_tokens"].shape[0]),
            "token_dim": int(self.train_dataset[0]["past_tokens"].shape[1]),
            "utilizer_config": self.utilizer_config,
            "variant_config": self.variant,
            "train_dataset_summary": self.train_dataset_summary,
            "test_dataset_summary": self.eval_dataset_summary,
        }
        checkpoint_path = self.checkpoint_dir / f"downstream_utilizer_step_{len(metrics):06d}.pt"
        summary["checkpoint_path"] = str(save_downstream_checkpoint(checkpoint_path, self.utilizer, self.optimizer, summary))
        eval_summary = {
            "phase": self.phase,
            "variant": self.variant["name"],
            "seed": int(self.seed),
            "split": "test",
            "checkpoint_path": summary["checkpoint_path"],
            "dataset_size": len(self.eval_dataset),
            **eval_metrics,
        }
        gap_summary = {
            **eval_summary,
            "student_teacher_gap": float(eval_metrics["student_future_mse"]) - float(eval_metrics["teacher_mse"]),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.eval_summary_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
        self.gap_summary_path.write_text(json.dumps(gap_summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary


def train_downstream_utilization_variant(
    config: dict[str, Any],
    *,
    phase: str,
    variant: dict[str, Any],
    seed: int,
    max_steps: int,
    batch_size: int,
    num_workers: int,
    run_dir: str | Path,
) -> dict[str, Any]:
    return DownstreamUtilizationTrainer(
        config,
        phase=phase,
        variant=variant,
        seed=seed,
        max_steps=max_steps,
        batch_size=batch_size,
        num_workers=num_workers,
        run_dir=run_dir,
    ).train()


def write_failed_variant_summary(
    *,
    run_dir: str | Path,
    phase: str,
    variant: dict[str, Any],
    seed: int,
    error: str,
) -> dict[str, Any]:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "success": False,
        "phase": phase,
        "variant": variant.get("name"),
        "seed": int(seed),
        "selection_policy": variant.get("selection_policy"),
        "utilization_module": variant.get("utilization_module"),
        "topk": int(variant.get("topk", 16)),
        "error": error,
        "run_dir": str(path),
        "summary_path": str(path / "summary.json"),
    }
    (path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def metrics_are_finite(row: dict[str, Any]) -> bool:
    return all(
        _is_finite(row.get(field))
        for field in ("student_future_mse", "teacher_mse", "student_teacher_ratio")
    )
