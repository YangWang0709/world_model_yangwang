"""Train and summarize Step 8 Student world-model baselines."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.student_selector_dataset import StudentSelectorDataset, student_selector_collate_fn
from eval.eval_efficiency import token_retention_ratio
from eval.eval_teacher_student_gap import compute_teacher_student_gap
from models.selection_policies import (
    BaseSelectionPolicy,
    build_selection_policy,
    compute_selection_metrics,
)
from models.student_world_model import StudentWorldModel
from models.teacher_world_model import TeacherWorldModel
from models.token_compressor import TokenCompressor
from training.losses import future_latent_mse
from training.student_selector_trainer import set_seed
from training.teacher_trainer import grad_norm, load_checkpoint, resolve_device, target_from_future_tokens


SUMMARY_COLUMNS = [
    "policy",
    "seed",
    "final_loss",
    "student_future_mse",
    "teacher_future_mse",
    "student_teacher_ratio",
    "token_retention_ratio",
    "selected_top1_hit_rate",
    "selected_topk_hit_rate",
    "selected_key_coverage",
]


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


def _load_teacher(checkpoint_path: str | Path, device: torch.device) -> TeacherWorldModel:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model = TeacherWorldModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _model_configs_from_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    compressor_cfg = config["compressor"]
    student_cfg = config["student_world_model"]
    compressor_config = {
        "token_dim": int(compressor_cfg.get("token_dim", 768)),
        "latent_dim": int(compressor_cfg.get("latent_dim", 512)),
        "num_latents": int(compressor_cfg.get("num_latents", 16)),
        "hidden_dim": int(compressor_cfg.get("hidden_dim", 512)),
        "dropout": float(compressor_cfg.get("dropout", 0.0)),
        "compressor_type": str(compressor_cfg.get("compressor_type", "perceiver_like")),
        "num_heads": int(compressor_cfg.get("num_heads", 8)),
    }
    student_world_model_config = {
        "latent_dim": int(student_cfg.get("latent_dim", compressor_config["latent_dim"])),
        "hidden_dim": int(student_cfg.get("hidden_dim", 512)),
        "output_dim": int(student_cfg.get("output_dim", 768)),
        "num_layers": int(student_cfg.get("num_layers", 2)),
        "dropout": float(student_cfg.get("dropout", 0.0)),
        "pool": str(student_cfg.get("pool", "mean")),
    }
    return compressor_config, student_world_model_config


def evaluate_baseline_models(
    policy: BaseSelectionPolicy,
    compressor: TokenCompressor,
    student_world_model: StudentWorldModel,
    teacher: TeacherWorldModel,
    loader: DataLoader,
    device: torch.device,
    topk: int,
) -> dict[str, float]:
    compressor.eval()
    student_world_model.eval()
    teacher.eval()
    student_sse = 0.0
    teacher_sse = 0.0
    element_count = 0
    all_indices = []
    all_masks = []
    with torch.no_grad():
        for batch in loader:
            past_tokens = batch["past_tokens"].to(device)
            future_tokens = batch["future_tokens"].to(device)
            target = target_from_future_tokens(future_tokens)
            selection = policy.select(past_tokens, batch=batch, k=topk, device=device)
            compressed_latents = compressor(selection["selected_tokens"])
            student_pred = student_world_model(compressed_latents)
            teacher_pred = teacher(past_tokens)
            if student_pred.shape != target.shape:
                raise ValueError(f"Student prediction shape {tuple(student_pred.shape)} != target {tuple(target.shape)}")
            if teacher_pred.shape != target.shape:
                raise ValueError(f"Teacher prediction shape {tuple(teacher_pred.shape)} != target {tuple(target.shape)}")
            student_sse += float((student_pred - target).pow(2).sum().item())
            teacher_sse += float((teacher_pred - target).pow(2).sum().item())
            element_count += int(target.numel())
            all_indices.append(selection["selected_indices"].detach().cpu())
            all_masks.append(batch["key_token_mask"].float().cpu())

    selected_indices = torch.cat(all_indices, dim=0)
    key_mask = torch.cat(all_masks, dim=0)
    student_mse = student_sse / float(max(element_count, 1))
    teacher_mse = teacher_sse / float(max(element_count, 1))
    gap_metrics = compute_teacher_student_gap(teacher_mse, student_mse)
    selection_metrics = compute_selection_metrics(
        selected_indices,
        key_mask,
        num_tokens=int(key_mask.shape[1]),
    )
    return {
        "student_future_mse": student_mse,
        "teacher_future_mse": teacher_mse,
        **gap_metrics,
        **selection_metrics,
    }


def save_baseline_student_checkpoint(
    path: str | Path,
    compressor: TokenCompressor,
    student_world_model: StudentWorldModel,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    policy_name: str,
    seed: int,
    compressor_config: dict[str, Any],
    student_world_model_config: dict[str, Any],
    metrics_summary: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "policy": str(policy_name),
            "seed": int(seed),
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


class BaselineStudentWorldModelTrainer:
    """Train one baseline policy/seed run."""

    def __init__(
        self,
        config: dict[str, Any],
        policy_name: str,
        seed: int,
        run_dir: str | Path,
    ) -> None:
        self.config = config
        self.policy_name = policy_name
        self.seed = int(seed)
        set_seed(int(config.get("seed", 42)) + self.seed)

        train_cfg = config["training"]
        output_cfg = config["output"]
        self.device = resolve_device(str(train_cfg.get("device", "cuda_if_available")))
        self.max_steps = int(train_cfg.get("max_steps", 300))
        self.log_every = int(train_cfg.get("log_every", 1))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
        self.topk = int(config["selection"].get("topk", 4))
        self.save_checkpoints = bool(output_cfg.get("save_checkpoint", True))
        self.save_metrics = bool(output_cfg.get("save_metrics", True))
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.eval_summary_path = self.run_dir / "eval_summary.json"
        self.gap_summary_path = self.run_dir / "teacher_student_gap_summary.json"

        self.dataset = _dataset_from_config(config)
        generator = torch.Generator()
        generator.manual_seed(int(config.get("seed", 42)) + self.seed)
        self.loader = DataLoader(
            self.dataset,
            batch_size=int(train_cfg.get("batch_size", 8)),
            shuffle=True,
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=student_selector_collate_fn,
            drop_last=False,
            generator=generator,
        )
        self.eval_loader = DataLoader(
            self.dataset,
            batch_size=int(train_cfg.get("batch_size", 8)),
            shuffle=False,
            num_workers=0,
            collate_fn=student_selector_collate_fn,
            drop_last=False,
        )

        self.policy = build_selection_policy(
            policy_name,
            seed=self.seed,
            learned_selector_checkpoint=config["learned_selector"]["checkpoint"],
        )
        self.teacher = _load_teacher(config["teacher_reference"]["checkpoint"], self.device)
        self.compressor_config, self.student_world_model_config = _model_configs_from_config(config)
        self.compressor = TokenCompressor(**self.compressor_config).to(self.device)
        self.student_world_model = StudentWorldModel(**self.student_world_model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.compressor.parameters()) + list(self.student_world_model.parameters()),
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
        while step < self.max_steps:
            self.compressor.train()
            self.student_world_model.train()
            for batch in self.loader:
                step += 1
                past_tokens = batch["past_tokens"].to(self.device)
                future_tokens = batch["future_tokens"].to(self.device)
                target = target_from_future_tokens(future_tokens)
                selection = self.policy.select(past_tokens, batch=batch, k=self.topk, device=self.device)
                compressed_latents = self.compressor(selection["selected_tokens"])
                pred = self.student_world_model(compressed_latents)
                if pred.shape != target.shape:
                    raise ValueError(f"Prediction shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
                loss = future_latent_mse(pred, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite baseline loss at step {step}: {loss.item()}")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm_before_clip = grad_norm(list(self.compressor.parameters()) + list(self.student_world_model.parameters()))
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.compressor.parameters()) + list(self.student_world_model.parameters()),
                        self.max_grad_norm,
                    )
                self.optimizer.step()

                selection_metrics = compute_selection_metrics(
                    selection["selected_indices"].detach().cpu(),
                    batch["key_token_mask"].float().cpu(),
                    num_tokens=int(past_tokens.shape[1]),
                )
                metric = {
                    "step": int(step),
                    "policy": self.policy_name,
                    "seed": int(self.seed),
                    "loss": float(loss.item()),
                    "grad_norm": float(norm_before_clip),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    **selection_metrics,
                }
                metrics.append(metric)
                if self.save_metrics:
                    with self.metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metric) + "\n")
                if self.log_every > 0 and step % self.log_every == 0:
                    print(
                        "policy={policy} seed={seed} step={step} loss={loss:.8f} "
                        "topk={topk:.3f} coverage={coverage:.3f}".format(
                            policy=self.policy_name,
                            seed=self.seed,
                            step=step,
                            loss=metric["loss"],
                            topk=metric["selected_topk_hit_rate"],
                            coverage=metric["selected_key_coverage"],
                        )
                    )
                if step >= self.max_steps:
                    break

        losses = [metric["loss"] for metric in metrics]
        eval_metrics = evaluate_baseline_models(
            self.policy,
            self.compressor,
            self.student_world_model,
            self.teacher,
            self.eval_loader,
            self.device,
            topk=self.topk,
        )
        summary = {
            "policy": self.policy_name,
            "seed": int(self.seed),
            "num_steps": len(metrics),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "loss_decreased": bool(losses[-1] < losses[0]),
            "student_future_mse": eval_metrics["student_future_mse"],
            "teacher_future_mse": eval_metrics["teacher_future_mse"],
            "student_teacher_gap": eval_metrics["student_teacher_gap"],
            "student_teacher_ratio": eval_metrics["student_teacher_ratio"],
            "token_retention_ratio": eval_metrics["token_retention_ratio"],
            "selected_top1_hit_rate": eval_metrics["selected_top1_hit_rate"],
            "selected_topk_hit_rate": eval_metrics["selected_topk_hit_rate"],
            "selected_key_coverage": eval_metrics["selected_key_coverage"],
            "selected_key_fraction": eval_metrics["selected_key_fraction"],
            "checkpoint_path": "",
            "metrics_path": str(self.metrics_path),
            "summary_path": str(self.summary_path),
            "eval_summary_path": str(self.eval_summary_path),
            "gap_summary_path": str(self.gap_summary_path),
            "device": str(self.device),
            "dataset_size": len(self.dataset),
            "run_dir": str(self.run_dir),
            "topk": int(self.topk),
            "total_tokens": int(self.dataset[0]["past_tokens"].shape[0]),
            "compressed_tokens": int(self.compressor_config["num_latents"]),
        }
        checkpoint_path = self.checkpoint_dir / f"student_world_model_step_{len(metrics):06d}.pt"
        if self.save_checkpoints:
            summary["checkpoint_path"] = str(checkpoint_path)
            save_baseline_student_checkpoint(
                checkpoint_path,
                compressor=self.compressor,
                student_world_model=self.student_world_model,
                optimizer=self.optimizer,
                step=len(metrics),
                policy_name=self.policy_name,
                seed=self.seed,
                compressor_config=self.compressor_config,
                student_world_model_config=self.student_world_model_config,
                metrics_summary=summary,
            )

        eval_summary = {
            "policy": self.policy_name,
            "seed": int(self.seed),
            "checkpoint_path": summary["checkpoint_path"],
            "dataset_size": len(self.dataset),
            "device": str(self.device),
            **eval_metrics,
        }
        gap_summary = dict(eval_summary)
        gap_summary["teacher_checkpoint_path"] = str(self.config["teacher_reference"]["checkpoint"])
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.eval_summary_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
        self.gap_summary_path.write_text(json.dumps(gap_summary, indent=2), encoding="utf-8")
        return summary


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def aggregate_baseline_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    random_rows = [row for row in rows if row["policy"] == "random_k"]
    learned_rows = [row for row in rows if row["policy"] == "learned_selector"]
    oracle_rows = [row for row in rows if row["policy"] == "oracle_key"]
    learned = learned_rows[0] if learned_rows else None
    oracle = oracle_rows[0] if oracle_rows else None
    random_mse = _mean_std([float(row["student_future_mse"]) for row in random_rows])
    random_coverage = _mean_std([float(row["selected_key_coverage"]) for row in random_rows])

    learned_vs_random = {}
    learned_vs_oracle = {}
    sanity_checks = {
        "learned_selector_present": learned is not None,
        "random_k_present": bool(random_rows),
        "oracle_key_present": oracle is not None,
        "learned_topk_hit_rate": False,
        "learned_key_coverage": False,
        "learned_future_mse_le_random_mean": False,
        "learned_ratio_finite": False,
    }
    if learned is not None:
        learned_vs_random = {
            "student_future_mse_delta": float(learned["student_future_mse"]) - random_mse["mean"],
            "selected_key_coverage_delta": float(learned["selected_key_coverage"]) - random_coverage["mean"],
            "future_mse_lower_than_random_mean": float(learned["student_future_mse"]) <= random_mse["mean"],
            "coverage_higher_than_random_mean": float(learned["selected_key_coverage"]) >= random_coverage["mean"],
        }
        sanity_checks["learned_topk_hit_rate"] = float(learned["selected_topk_hit_rate"]) >= 0.8
        sanity_checks["learned_key_coverage"] = float(learned["selected_key_coverage"]) >= 0.8
        sanity_checks["learned_future_mse_le_random_mean"] = float(learned["student_future_mse"]) <= random_mse["mean"]
        sanity_checks["learned_ratio_finite"] = _finite(float(learned["student_teacher_ratio"]))
        if oracle is not None:
            learned_vs_oracle = {
                "student_future_mse_gap": float(learned["student_future_mse"]) - float(oracle["student_future_mse"]),
                "selected_key_coverage_gap": float(learned["selected_key_coverage"]) - float(oracle["selected_key_coverage"]),
            }

    return {
        "random_k_student_future_mse": random_mse,
        "random_k_selected_key_coverage": random_coverage,
        "learned_vs_random": learned_vs_random,
        "learned_vs_oracle": learned_vs_oracle,
        "sanity_gate": {
            "checks": sanity_checks,
            "pass": all(sanity_checks.values()),
        },
    }


def render_baseline_markdown(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    lines = [
        "| policy | seed | final_loss | student_future_mse | teacher_future_mse | student_teacher_ratio | token_retention_ratio | selected_top1_hit_rate | selected_topk_hit_rate | selected_key_coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (str(item["policy"]), int(item["seed"]))):
        lines.append(
            "| {policy} | {seed} | {final_loss:.8f} | {student_future_mse:.8f} | "
            "{teacher_future_mse:.8f} | {student_teacher_ratio:.6f} | {token_retention_ratio:.6f} | "
            "{selected_top1_hit_rate:.6f} | {selected_topk_hit_rate:.6f} | {selected_key_coverage:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "```json",
            json.dumps(aggregate, indent=2),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def write_baseline_summaries(
    rows: list[dict[str, Any]],
    run_dir: str | Path,
    output_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    output_cfg = output_cfg or {}
    json_path = Path(output_cfg.get("summary_json", run_path / "baseline_summary.json"))
    csv_path = Path(output_cfg.get("summary_csv", run_path / "baseline_summary.csv"))
    md_path = Path(output_cfg.get("summary_md", run_path / "baseline_summary.md"))
    for path in (json_path, csv_path, md_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    aggregate = aggregate_baseline_rows(rows)
    payload = {
        "run_dir": str(run_path),
        "rows": rows,
        "aggregate": aggregate,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (str(item["policy"]), int(item["seed"]))):
            writer.writerow({column: row[column] for column in SUMMARY_COLUMNS})
    md_path.write_text(render_baseline_markdown(rows, aggregate), encoding="utf-8")
    return {
        "summary_json": str(json_path),
        "summary_csv": str(csv_path),
        "summary_md": str(md_path),
        "aggregate": aggregate,
    }


def train_baseline_comparison(config: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    policies = list(config["selection"].get("policies", []))
    repeat_seeds = config["training"].get("repeat_random_seeds", {})
    rows = []
    for policy_name in policies:
        seeds = repeat_seeds.get(policy_name, [0])
        for seed in seeds:
            subrun_dir = run_dir / f"{policy_name}_seed{int(seed)}"
            trainer = BaselineStudentWorldModelTrainer(
                config=config,
                policy_name=policy_name,
                seed=int(seed),
                run_dir=subrun_dir,
            )
            summary = trainer.train()
            rows.append(summary)
    summary_paths = write_baseline_summaries(rows, run_dir=run_dir, output_cfg=config.get("output", {}))
    result = {
        "run_dir": str(run_dir),
        "policies": policies,
        "rows": rows,
        **summary_paths,
    }
    print(json.dumps(result, indent=2))
    return result
