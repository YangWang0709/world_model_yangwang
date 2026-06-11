"""Train and evaluate Step17 context bottleneck world-model variants."""

from __future__ import annotations

import argparse
import csv
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

from data.context_token_shard_dataset import ContextSelectorDataset, context_selector_collate_fn, summarize_context_dataset
from models.context_bottleneck_world_model import ContextBottleneckWorldModel, ContextTeacherWorldModel, future_target_from_tokens
from models.context_selection_policies import context_selection_metrics, select_context_tokens, topk_from_retention
from training.losses import future_latent_mse
from training.teacher_trainer import grad_norm, resolve_device
from training.train_context_teacher import load_context_teacher_checkpoint
from training.train_unified_context_selector import load_unified_selector_checkpoint


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


def save_context_bottleneck_checkpoint(path: str | Path, model: ContextBottleneckWorldModel, optimizer: torch.optim.Optimizer | None, step: int, model_config: dict[str, Any], summary: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model_config": dict(model_config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics_summary": summary,
            "current_tokens_dropped": False,
        },
        out,
    )
    return out


def _zero_context_selection(context_tokens: torch.Tensor) -> dict[str, Any]:
    zeros = context_tokens[:, :1].clone()
    zeros.zero_()
    return {
        "selected_tokens": zeros,
        "selected_indices": torch.zeros(context_tokens.shape[0], 1, dtype=torch.long, device=context_tokens.device),
        "selected_scores": None,
        "scores": None,
        "policy_name": "current_only",
    }


def _select_for_policy(
    *,
    context_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    batch: dict[str, Any],
    policy: str,
    topk: int,
    selector: Any,
    seed: int,
) -> dict[str, Any]:
    if policy == "current_only":
        return _zero_context_selection(context_tokens)
    return select_context_tokens(
        context_tokens=context_tokens,
        current_tokens=current_tokens,
        policy=policy,
        topk=topk,
        selector=selector,
        batch=batch,
        seed=seed,
        learned_k=topk // 2,
        uniform_k=topk - (topk // 2),
    )


def evaluate_context_bottleneck_model(
    model: ContextBottleneckWorldModel,
    loader: DataLoader,
    device: torch.device,
    *,
    policy: str,
    topk: int,
    selector: Any = None,
    teacher: ContextTeacherWorldModel | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    model.eval()
    if selector is not None:
        selector.eval()
    sse = 0.0
    teacher_sse = 0.0
    count = 0
    indices_all = []
    importance_all = []
    with torch.no_grad():
        for batch in loader:
            context = batch["context_tokens"].to(device)
            current = batch["current_tokens"].to(device)
            future = batch["future_tokens"].to(device)
            target = future_target_from_tokens(future)
            selection = _select_for_policy(context_tokens=context, current_tokens=current, batch=batch, policy=policy, topk=topk, selector=selector, seed=seed)
            pred = model(current, selection["selected_tokens"], selection.get("selected_scores"))
            sse += float((pred - target).pow(2).sum().item())
            if teacher is not None:
                teacher_pred = teacher(context, current)
                teacher_sse += float((teacher_pred - target).pow(2).sum().item())
            count += int(target.numel())
            if selection["selected_indices"].shape[1] > 0:
                indices_all.append(selection["selected_indices"].detach().cpu())
                importance_all.append(batch["importance_scores_norm"].float().cpu())
    mse = sse / float(max(count, 1))
    teacher_mse = teacher_sse / float(max(count, 1)) if teacher is not None else None
    metrics: dict[str, Any] = {
        "future_mse": mse,
        "student_future_mse": mse,
        "context_teacher_mse": teacher_mse,
        "student_teacher_ratio": mse / max(float(teacher_mse), 1e-12) if teacher_mse is not None else None,
        "current_tokens_dropped": False,
    }
    if indices_all:
        metrics.update(
            context_selection_metrics(
                torch.cat(indices_all, dim=0),
                torch.cat(importance_all, dim=0),
                num_tokens=int(torch.cat(importance_all, dim=0).shape[1]),
                seed=seed,
            )
        )
    return metrics


class ContextBottleneckTrainer:
    def __init__(self, config: dict[str, Any], *, policy: str | None = None, seed: int | None = None, run_dir: str | Path | None = None) -> None:
        self.config = config
        self.policy = str(policy or config.get("selection", {}).get("policy", "learned_context_selector_topK"))
        self.seed = int(seed if seed is not None else config.get("seed", 42))
        set_seed(self.seed)
        self.device = resolve_device(str(config["training"].get("device", "cuda_if_available")))
        self.run_dir = Path(run_dir) if run_dir is not None else Path(config["output"]["run_root"]) / config["output"]["run_name"]
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.train_dataset = _dataset(config, "train")
        self.test_dataset = _dataset(config, "test")
        self.train_summary = summarize_context_dataset(self.train_dataset, "train")
        self.test_summary = summarize_context_dataset(self.test_dataset, "test")
        self.topk = topk_from_retention(
            int(self.train_summary["num_context_tokens"]),
            float(config.get("selection", {}).get("retention_ratio", 0.04081632653061224)),
            int(config.get("selection", {}).get("fallback_topk_if_unknown", 32)),
        )
        self.loader = DataLoader(self.train_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=True, num_workers=int(config["training"].get("num_workers", 0)), collate_fn=context_selector_collate_fn)
        self.eval_loader = DataLoader(self.test_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0, collate_fn=context_selector_collate_fn)
        self.model_config = dict(config["model"])
        self.model = ContextBottleneckWorldModel(**self.model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
        self.max_steps = int(config["training"]["max_steps"])
        self.max_grad_norm = float(config["training"].get("max_grad_norm", 0.0))
        self.selector = None
        if self.policy in {"learned_context_selector_topK", "hybrid_context_learned_uniform"}:
            self.selector, _ = load_unified_selector_checkpoint(config["selector"]["checkpoint"], self.device)
            for parameter in self.selector.parameters():
                parameter.requires_grad = False
        self.teacher, _ = load_context_teacher_checkpoint(config["teacher_reference"]["checkpoint"], self.device)
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

    def train(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text("", encoding="utf-8")
        rows: list[dict[str, Any]] = []
        step = 0
        while step < self.max_steps:
            self.model.train()
            for batch in self.loader:
                step += 1
                context = batch["context_tokens"].to(self.device)
                current = batch["current_tokens"].to(self.device)
                target = future_target_from_tokens(batch["future_tokens"].to(self.device))
                selection = _select_for_policy(context_tokens=context, current_tokens=current, batch=batch, policy=self.policy, topk=self.topk, selector=self.selector, seed=self.seed)
                pred = self.model(current, selection["selected_tokens"], selection.get("selected_scores"))
                loss = future_latent_mse(pred, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite context bottleneck loss at step {step}")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = grad_norm(self.model.parameters())
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                row = {"step": step, "loss": float(loss.item()), "grad_norm": float(norm), "policy": self.policy, "seed": self.seed}
                rows.append(row)
                with self.metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                if step >= self.max_steps:
                    break
        losses = [row["loss"] for row in rows]
        eval_metrics = evaluate_context_bottleneck_model(
            self.model,
            self.eval_loader,
            self.device,
            policy=self.policy,
            topk=self.topk,
            selector=self.selector,
            teacher=self.teacher,
            seed=self.seed,
        )
        summary = {
            "success": True,
            "policy": self.policy,
            "seed": self.seed,
            "num_steps": len(rows),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "loss_decreased": bool(losses[-1] < losses[0]),
            "checkpoint_path": "",
            "metrics_path": str(self.metrics_path),
            "summary_path": str(self.summary_path),
            "run_dir": str(self.run_dir),
            "context_topK": int(self.topk),
            "context_retention_ratio": float(self.topk) / float(self.train_summary["num_context_tokens"]),
            "current_tokens_dropped": False,
            "trained_current_importance": False,
            "trained_context_importance": True,
            "train_dataset_summary": self.train_summary,
            "test_dataset_summary": self.test_summary,
            **eval_metrics,
        }
        checkpoint_path = self.checkpoint_dir / f"context_bottleneck_world_model_step_{len(rows):06d}.pt"
        summary["checkpoint_path"] = str(save_context_bottleneck_checkpoint(checkpoint_path, self.model, self.optimizer, len(rows), self.model_config, summary))
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.run_dir / "eval_summary.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary


def train_context_bottleneck_world_model(config: dict[str, Any]) -> dict[str, Any]:
    return ContextBottleneckTrainer(config).train()


def _read_rows(root: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*/summary.json"))]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("success", True):
            grouped.setdefault(str(row["policy"]), []).append(row)
    agg_rows = []
    for policy, items in sorted(grouped.items()):
        values = [float(item["student_future_mse"]) for item in items if item.get("student_future_mse") is not None]
        if not values:
            continue
        agg_rows.append(
            {
                "policy": policy,
                "seeds": ",".join(str(item.get("seed")) for item in items),
                "n": len(values),
                "student_future_mse_mean": float(torch.tensor(values).mean().item()),
                "student_future_mse_std": float(torch.tensor(values).std(unbiased=False).item()) if len(values) > 1 else 0.0,
                "selected_context_importance_mean": float(torch.tensor([float(item.get("selected_context_importance_mean") or 0.0) for item in items]).mean().item()),
                "selector_target_topk_overlap_mean": float(torch.tensor([float(item.get("selector_target_topk_overlap") or 0.0) for item in items]).mean().item()),
            }
        )
    return {"rows": rows, "aggregate": agg_rows}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["policy", "seeds", "n", "student_future_mse_mean", "student_future_mse_std", "selected_context_importance_mean", "selector_target_topk_overlap_mean"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def run_context_bottleneck_baselines(config: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    policies = list(config["baselines"]["policies"])
    for policy in policies:
        if policy == "full_context_teacher_reference":
            continue
        if policy == "random_context_topK":
            seeds = [int(seed) for seed in config["baselines"].get("random_seeds", [0, 1, 2])]
        elif policy in {"learned_context_selector_topK", "hybrid_context_learned_uniform"}:
            seeds = [int(seed) for seed in config["baselines"].get("learned_seeds", [0, 1, 2])]
        elif policy == "uniform_context_topK":
            seeds = [int(seed) for seed in config["baselines"].get("uniform_seeds", [0])]
        elif policy == "teacher_context_importance_topK":
            seeds = [int(seed) for seed in config["baselines"].get("oracle_seeds", [0])]
        else:
            seeds = [int(seed) for seed in config["baselines"].get("current_only_seeds", [0])]
        for seed in seeds:
            out = run_dir / f"{policy}_seed{seed}"
            if (out / "summary.json").exists():
                continue
            ContextBottleneckTrainer(config, policy=policy, seed=seed, run_dir=out).train()
    rows = _read_rows(run_dir)
    if "full_context_teacher_reference" in policies and not any(row.get("policy") == "full_context_teacher_reference" for row in rows):
        teacher_mse = next((row.get("context_teacher_mse") for row in rows if row.get("context_teacher_mse") is not None), None)
        if teacher_mse is not None:
            teacher_dir = run_dir / "full_context_teacher_reference_seed0"
            teacher_dir.mkdir(parents=True, exist_ok=True)
            teacher_row = {
                "success": True,
                "policy": "full_context_teacher_reference",
                "seed": 0,
                "num_steps": 0,
                "student_future_mse": float(teacher_mse),
                "future_mse": float(teacher_mse),
                "context_teacher_mse": float(teacher_mse),
                "student_teacher_ratio": 1.0,
                "selected_context_importance_mean": None,
                "selector_target_topk_overlap": None,
                "current_tokens_dropped": False,
                "run_dir": str(teacher_dir),
                "summary_path": str(teacher_dir / "summary.json"),
            }
            (teacher_dir / "summary.json").write_text(json.dumps(teacher_row, indent=2), encoding="utf-8")
            rows.append(teacher_row)
    aggregate = _aggregate(rows)
    (run_dir / "baseline_summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    (run_dir / "baseline_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    _write_csv(run_dir / "baseline_aggregate.csv", aggregate["aggregate"])
    lines = ["# Step17 Context Bottleneck Baseline Aggregate", "", "| policy | seeds | n | mse_mean | mse_std | selected_importance | topK_overlap |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(aggregate["aggregate"], key=lambda item: float(item["student_future_mse_mean"])):
        lines.append(f"| {row['policy']} | {row['seeds']} | {row['n']} | {row['student_future_mse_mean']:.8f} | {row['student_future_mse_std']:.8f} | {row['selected_context_importance_mean']:.6f} | {row['selector_target_topk_overlap_mean']:.6f} |")
    (run_dir / "baseline_aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "baseline_summary.csv").write_text((run_dir / "baseline_aggregate.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "baseline_summary.md").write_text((run_dir / "baseline_aggregate.md").read_text(encoding="utf-8"), encoding="utf-8")
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "train_context_bottleneck_world_model_bair_1000_128.yaml"))
    parser.add_argument("--baselines", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.baselines:
        run_context_bottleneck_baselines(config)
    else:
        train_context_bottleneck_world_model(config)


if __name__ == "__main__":
    main()
