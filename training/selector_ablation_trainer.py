"""Train and summarize BAIR selector loss ablations."""

from __future__ import annotations

import copy
import csv
import gc
import json
import math
import statistics
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.student_selector_dataset import StudentSelectorDataset, student_selector_collate_fn
from models.attention_selector import AttentionSelector
from models.selector_losses import compute_selector_loss
from training.baseline_student_world_model_trainer import BaselineStudentWorldModelTrainer
from training.losses import importance_topk_metrics
from training.student_selector_trainer import (
    evaluate_selector_on_loader,
    save_student_selector_checkpoint,
    set_seed,
)
from training.teacher_trainer import grad_norm, resolve_device


SELECTOR_SUMMARY_COLUMNS = [
    "variant_name",
    "loss_type",
    "seed",
    "final_loss",
    "test_importance_mse",
    "test_importance_mae",
    "test_pearson_corr_mean",
    "test_target_top1_overlap",
    "test_target_topk_overlap",
    "test_selected_teacher_importance_mean",
    "test_selected_vs_random_importance_gap",
]

DOWNSTREAM_SUMMARY_COLUMNS = [
    "variant_name",
    "loss_type",
    "seed",
    "final_loss",
    "student_future_mse",
    "teacher_mse",
    "student_teacher_ratio",
    "token_retention_ratio",
    "selector_target_top1_overlap",
    "selector_target_topk_overlap",
    "selected_teacher_importance_mean",
    "selected_vs_random_importance_gap",
]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt(value: Any, precision: int = 8) -> str:
    if value is None:
        return "n/a"
    if _finite(value):
        return f"{float(value):.{precision}f}"
    return str(value)


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if _finite(row.get(key))]


def _variant_run_name(variant: dict[str, Any]) -> str:
    return f"{variant['name']}_seed{int(variant.get('seed', 0))}"


def _dataset_from_config(config: dict[str, Any], split: str) -> StudentSelectorDataset:
    data_cfg = config["data"]
    if split == "train":
        token_shard_dir = data_cfg.get("train_token_shard_dir", data_cfg.get("token_shard_dir"))
        importance_shard_dir = data_cfg.get("train_importance_shard_dir", data_cfg.get("importance_shard_dir"))
        max_samples = data_cfg.get("max_train_samples", data_cfg.get("max_samples"))
    elif split == "test":
        token_shard_dir = data_cfg.get("test_token_shard_dir", data_cfg.get("token_shard_dir"))
        importance_shard_dir = data_cfg.get("test_importance_shard_dir", data_cfg.get("importance_shard_dir"))
        max_samples = data_cfg.get("max_test_samples", data_cfg.get("max_samples"))
    else:
        raise ValueError(f"Unsupported split: {split!r}")
    if token_shard_dir is None or importance_shard_dir is None:
        raise KeyError(f"Missing token/importance shard dirs for split {split!r}")
    return StudentSelectorDataset(
        token_shard_dir=token_shard_dir,
        importance_shard_dir=importance_shard_dir,
        token_shard_glob=data_cfg.get("token_shard_glob", "tokens_shard_*.pt"),
        importance_shard_glob=data_cfg.get("importance_shard_glob", "importance_shard_*.pt"),
        require_key_token_mask=bool(data_cfg.get("require_key_token_mask", False)),
        map_location="cpu",
        max_samples=int(max_samples) if max_samples is not None else None,
    )


class SelectorVariantTrainer:
    """Fit one frozen-checkpoint selector variant against Teacher importance."""

    def __init__(self, config: dict[str, Any], variant: dict[str, Any], run_dir: str | Path) -> None:
        self.config = config
        self.variant = dict(variant)
        self.variant_name = str(self.variant["name"])
        self.loss_type = str(self.variant.get("type", self.variant_name))
        self.seed = int(self.variant.get("seed", 0))
        set_seed(int(config.get("seed", 42)) + self.seed)

        train_cfg = config["selector_training"]
        model_cfg = config["selector_model"]
        selection_cfg = config.get("selection", {})
        output_cfg = config.get("output", {})

        self.device = resolve_device(str(train_cfg.get("device", "cuda_if_available")))
        self.max_steps = int(train_cfg.get("max_steps", 300))
        self.log_every = int(train_cfg.get("log_every", 100))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.topk = int(self.variant.get("topk", selection_cfg.get("topk", 16)))
        self.save_checkpoints = bool(output_cfg.get("save_checkpoint", True))
        self.save_metrics = bool(output_cfg.get("save_metrics", True))

        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.eval_summary_path = self.run_dir / "eval_summary.json"

        self.train_dataset = _dataset_from_config(config, split="train")
        self.test_dataset = _dataset_from_config(config, split="test")
        generator = torch.Generator()
        generator.manual_seed(int(config.get("seed", 42)) + self.seed)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(train_cfg.get("batch_size", 4)),
            shuffle=True,
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=student_selector_collate_fn,
            drop_last=False,
            generator=generator,
        )
        self.train_eval_loader = DataLoader(
            self.train_dataset,
            batch_size=int(train_cfg.get("batch_size", 4)),
            shuffle=False,
            num_workers=0,
            collate_fn=student_selector_collate_fn,
            drop_last=False,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=int(train_cfg.get("batch_size", 4)),
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
        while step < self.max_steps:
            self.model.train()
            for batch in self.train_loader:
                step += 1
                past_tokens = batch["past_tokens"].to(self.device)
                targets = batch["importance_scores_norm"].to(self.device)
                logits = self.model(past_tokens)
                loss_parts = compute_selector_loss(logits, targets, self.variant)
                loss = loss_parts["loss"]

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm_before_clip = grad_norm(self.model.parameters())
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    score_probs = torch.sigmoid(logits.detach()).cpu()
                    metric_values = importance_topk_metrics(
                        score_probs,
                        targets.detach().cpu(),
                        k=self.topk,
                    )
                metric = {
                    "step": int(step),
                    "variant_name": self.variant_name,
                    "loss_type": self.loss_type,
                    "loss": float(loss.item()),
                    "mse_loss": float(loss_parts["mse_loss"].item()),
                    "weighted_mse_loss": float(loss_parts["weighted_mse_loss"].item()),
                    "rank_loss": float(loss_parts["rank_loss"].item()),
                    "topk_bce_loss": float(loss_parts["topk_bce_loss"].item()),
                    "grad_norm": float(norm_before_clip),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    **metric_values,
                }
                metrics.append(metric)
                if self.save_metrics:
                    with self.metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metric) + "\n")
                if self.log_every > 0 and step % self.log_every == 0:
                    print(
                        "selector_variant={variant} step={step} loss={loss:.8f} "
                        "mse={mse:.8f} rank={rank:.8f} bce={bce:.8f} topk={topk:.4f}".format(
                            variant=self.variant_name,
                            step=step,
                            loss=metric["loss"],
                            mse=metric["mse_loss"],
                            rank=metric["rank_loss"],
                            bce=metric["topk_bce_loss"],
                            topk=metric["target_topk_overlap"],
                        )
                    )
                if step >= self.max_steps:
                    break

        losses = [metric["loss"] for metric in metrics]
        train_eval = evaluate_selector_on_loader(self.model, self.train_eval_loader, self.device, self.topk)
        test_eval = evaluate_selector_on_loader(self.model, self.test_loader, self.device, self.topk)
        checkpoint_path = self.checkpoint_dir / f"student_selector_step_{len(metrics):06d}.pt"
        summary = {
            "variant_name": self.variant_name,
            "variant_run_name": self.run_dir.name,
            "loss_type": self.loss_type,
            "seed": int(self.seed),
            "success": True,
            "num_steps": len(metrics),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "loss_decreased": bool(losses[-1] < losses[0]),
            "test_importance_mse": test_eval["importance_mse"],
            "test_importance_mae": test_eval["importance_mae"],
            "test_pearson_corr_mean": test_eval["pearson_corr_mean"],
            "test_target_top1_overlap": test_eval["target_top1_overlap"],
            "test_target_topk_overlap": test_eval["target_topk_overlap"],
            "test_selected_teacher_importance_mean": test_eval["selected_teacher_importance_mean"],
            "test_random_teacher_importance_mean": test_eval["random_teacher_importance_mean"],
            "test_selected_vs_random_importance_gap": test_eval["selected_vs_random_importance_gap"],
            "score_mean": test_eval["score_mean"],
            "score_std": test_eval["score_std"],
            "target_mean": test_eval["target_mean"],
            "target_std": test_eval["target_std"],
            "checkpoint_path": str(checkpoint_path) if self.save_checkpoints else "",
            "metrics_path": str(self.metrics_path),
            "summary_path": str(self.summary_path),
            "eval_summary_path": str(self.eval_summary_path),
            "run_dir": str(self.run_dir),
            "device": str(self.device),
            "topk": int(self.topk),
            "train_num_samples": len(self.train_dataset),
            "test_num_samples": len(self.test_dataset),
            "num_tokens": int(self.train_dataset[0]["past_tokens"].shape[0]),
            "token_dim": int(self.train_dataset[0]["past_tokens"].shape[1]),
            "loss_config": dict(self.variant),
            "train_eval_metrics": train_eval,
            "test_eval_metrics": test_eval,
        }
        if self.save_checkpoints:
            save_student_selector_checkpoint(
                checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                step=len(metrics),
                model_config=self.model_config,
                metrics_summary=summary,
            )
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.eval_summary_path.write_text(json.dumps(test_eval, indent=2), encoding="utf-8")
        return summary


def train_selector_variants(config: dict[str, Any], run_dir: str | Path | None = None) -> list[dict[str, Any]]:
    base_run_dir = Path(run_dir or Path(config["output"]["run_root"]) / config["output"]["run_name"])
    selector_root = base_run_dir / "selectors"
    selector_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in config.get("loss_variants", []):
        subrun = selector_root / _variant_run_name(variant)
        try:
            trainer = SelectorVariantTrainer(config, variant, subrun)
            rows.append(trainer.train())
        except Exception as error:  # pragma: no cover - exercised by smoke failure path
            subrun.mkdir(parents=True, exist_ok=True)
            failure = {
                "variant_name": str(variant.get("name", "unknown")),
                "variant_run_name": _variant_run_name(variant) if "name" in variant else subrun.name,
                "loss_type": str(variant.get("type", variant.get("name", "unknown"))),
                "seed": int(variant.get("seed", 0)),
                "success": False,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "run_dir": str(subrun),
                "summary_path": str(subrun / "summary.json"),
            }
            (subrun / "summary.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            rows.append(failure)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def _downstream_config(config: dict[str, Any], selector_summary: dict[str, Any], run_root: Path) -> dict[str, Any]:
    training_cfg = copy.deepcopy(config["downstream_student_world_model"])
    training_cfg.setdefault("device", config.get("selector_training", {}).get("device", "cuda_if_available"))
    training_cfg.setdefault("log_every", 100)
    return {
        "stage": "selector_ablation_downstream_student_world_model",
        "project_root": config.get("project_root"),
        "seed": int(config.get("seed", 42)) + int(selector_summary.get("seed", 0)),
        "data": copy.deepcopy(config["data"]),
        "learned_selector": {
            "checkpoint": selector_summary["checkpoint_path"],
        },
        "selection": {
            "topk": int(config.get("selection", {}).get("topk", selector_summary.get("topk", 16))),
        },
        "teacher_reference": copy.deepcopy(config["teacher_reference"]),
        "compressor": copy.deepcopy(config["compressor"]),
        "student_world_model": copy.deepcopy(config["student_world_model"]),
        "training": training_cfg,
        "output": {
            "run_root": str(run_root),
            "run_name": str(selector_summary["variant_run_name"]),
            "save_checkpoint": bool(config.get("output", {}).get("save_checkpoint", True)),
            "save_metrics": bool(config.get("output", {}).get("save_metrics", True)),
        },
    }


def train_downstream_variants(
    config: dict[str, Any],
    selector_rows: list[dict[str, Any]],
    run_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    if not bool(config.get("downstream_student_world_model", {}).get("enabled", True)):
        return []
    base_run_dir = Path(run_dir or Path(config["output"]["run_root"]) / config["output"]["run_name"])
    downstream_root = base_run_dir / "downstream_student_world_models"
    downstream_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for selector_summary in selector_rows:
        if not selector_summary.get("success", False):
            continue
        subrun = downstream_root / selector_summary["variant_run_name"]
        try:
            downstream_cfg = _downstream_config(config, selector_summary, downstream_root)
            trainer = BaselineStudentWorldModelTrainer(
                config=downstream_cfg,
                policy_name="learned_selector",
                seed=int(selector_summary.get("seed", 0)),
                run_dir=subrun,
            )
            summary = trainer.train()
            summary.update(
                {
                    "variant_name": selector_summary["variant_name"],
                    "variant_run_name": selector_summary["variant_run_name"],
                    "loss_type": selector_summary["loss_type"],
                    "selector_checkpoint": selector_summary["checkpoint_path"],
                    "selector_checkpoint_path": selector_summary["checkpoint_path"],
                    "success": True,
                }
            )
            for path_key in ("summary_path", "eval_summary_path", "gap_summary_path"):
                path = summary.get(path_key)
                if path and Path(path).exists():
                    payload = json.loads(Path(path).read_text(encoding="utf-8"))
                    payload.update(
                        {
                            "variant_name": summary["variant_name"],
                            "variant_run_name": summary["variant_run_name"],
                            "loss_type": summary["loss_type"],
                            "selector_checkpoint": summary["selector_checkpoint"],
                        }
                    )
                    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            Path(summary["summary_path"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
            rows.append(summary)
        except Exception as error:  # pragma: no cover - exercised by smoke failure path
            subrun.mkdir(parents=True, exist_ok=True)
            failure = {
                "variant_name": selector_summary.get("variant_name", "unknown"),
                "variant_run_name": selector_summary.get("variant_run_name", subrun.name),
                "loss_type": selector_summary.get("loss_type", "unknown"),
                "seed": int(selector_summary.get("seed", 0)),
                "success": False,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "run_dir": str(subrun),
                "summary_path": str(subrun / "summary.json"),
            }
            (subrun / "summary.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            rows.append(failure)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def load_selector_ablation_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted((Path(run_dir) / "selectors").glob("*/summary.json")):
        rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(f"No selector summaries found under {Path(run_dir) / 'selectors'}")
    return rows


def load_downstream_ablation_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted((Path(run_dir) / "downstream_student_world_models").glob("*/summary.json")):
        rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(
            f"No downstream summaries found under {Path(run_dir) / 'downstream_student_world_models'}"
        )
    return rows


def aggregate_selector_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("success", True)]
    mse_only_ok = any(row.get("loss_type") == "mse_only" for row in successful)
    non_mse_ok = any(row.get("loss_type") != "mse_only" for row in successful)
    finite_rows = [
        row
        for row in successful
        if _finite(row.get("test_selected_teacher_importance_mean"))
        and _finite(row.get("test_target_topk_overlap"))
        and _finite(row.get("test_importance_mse"))
    ]
    best_importance = (
        max(finite_rows, key=lambda row: float(row["test_selected_teacher_importance_mean"])) if finite_rows else None
    )
    best_topk = max(finite_rows, key=lambda row: float(row["test_target_topk_overlap"])) if finite_rows else None
    best_mse = min(finite_rows, key=lambda row: float(row["test_importance_mse"])) if finite_rows else None
    checks = {
        "selector_metrics_finite": len(finite_rows) == len(successful) and bool(successful),
        "mse_only_variant_success": mse_only_ok,
        "non_mse_only_variant_success": non_mse_ok,
        "best_selector_importance_finite": best_importance is not None,
    }
    return {
        "num_variants": len(rows),
        "num_successful": len(successful),
        "failed_variants": [row for row in rows if not row.get("success", True)],
        "best_by_selected_teacher_importance": best_importance,
        "best_by_target_topk_overlap": best_topk,
        "best_by_importance_mse": best_mse,
        "mean_selected_teacher_importance": _mean(
            _numeric_values(successful, "test_selected_teacher_importance_mean")
        ),
        "mean_target_topk_overlap": _mean(_numeric_values(successful, "test_target_topk_overlap")),
        "sanity_gate": {"checks": checks, "pass": all(checks.values())},
    }


def _step12_rows(step12_summary_json: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(step12_summary_json).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported Step 12 summary format: {step12_summary_json}")


def build_step12_reference(step12_summary_json: str | Path) -> dict[str, Any]:
    rows = _step12_rows(step12_summary_json)
    refs: dict[str, Any] = {"summary_json": str(step12_summary_json)}
    random_rows = [row for row in rows if row.get("policy") == "random_k"]
    refs["random_k_mean"] = {
        "method": "step12_random_k_mean",
        "student_future_mse": _mean(_numeric_values(random_rows, "student_future_mse")),
        "selected_teacher_importance_mean": _mean(
            _numeric_values(random_rows, "selected_teacher_importance_mean")
        ),
        "selector_target_topk_overlap": _mean(_numeric_values(random_rows, "selector_target_topk_overlap")),
    }
    for policy in ("uniform_k", "teacher_importance_topk", "learned_selector"):
        match = next((row for row in rows if row.get("policy") == policy), None)
        refs[policy] = match
    return refs


def aggregate_downstream_rows(
    rows: list[dict[str, Any]],
    step12_summary_json: str | Path | None = None,
) -> dict[str, Any]:
    successful = [row for row in rows if row.get("success", True)]
    finite_rows = [row for row in successful if _finite(row.get("student_future_mse"))]
    best_mse = min(finite_rows, key=lambda row: float(row["student_future_mse"])) if finite_rows else None
    checks = {
        "downstream_result_present": bool(finite_rows),
        "downstream_metrics_finite": len(finite_rows) == len(successful) and bool(successful),
        "best_downstream_mse_finite": best_mse is not None,
    }
    step12_refs: dict[str, Any] = {}
    comparison: dict[str, Any] = {}
    if step12_summary_json is not None and Path(step12_summary_json).exists():
        step12_refs = build_step12_reference(step12_summary_json)
        learned = step12_refs.get("learned_selector") or {}
        uniform = step12_refs.get("uniform_k") or {}
        teacher_topk = step12_refs.get("teacher_importance_topk") or {}
        learned_mse = learned.get("student_future_mse")
        learned_importance = learned.get("selected_teacher_importance_mean")
        learned_topk = learned.get("selector_target_topk_overlap")
        comparison = {
            "downstream_improvement_over_step12_learned": (
                best_mse is not None
                and _finite(learned_mse)
                and float(best_mse["student_future_mse"]) < float(learned_mse)
            ),
            "best_downstream_vs_step12_learned_mse_delta": (
                float(best_mse["student_future_mse"]) - float(learned_mse)
                if best_mse is not None and _finite(learned_mse)
                else None
            ),
            "any_variant_selected_importance_gt_step12_learned": (
                any(
                    _finite(row.get("selected_teacher_importance_mean"))
                    and _finite(learned_importance)
                    and float(row["selected_teacher_importance_mean"]) > float(learned_importance)
                    for row in successful
                )
            ),
            "any_variant_topk_overlap_gt_step12_learned": (
                any(
                    _finite(row.get("selector_target_topk_overlap"))
                    and _finite(learned_topk)
                    and float(row["selector_target_topk_overlap"]) > float(learned_topk)
                    for row in successful
                )
            ),
            "best_downstream_beats_step12_uniform_mse": (
                best_mse is not None
                and _finite(uniform.get("student_future_mse"))
                and float(best_mse["student_future_mse"]) < float(uniform["student_future_mse"])
            ),
            "best_downstream_vs_step12_uniform_mse_delta": (
                float(best_mse["student_future_mse"]) - float(uniform["student_future_mse"])
                if best_mse is not None and _finite(uniform.get("student_future_mse"))
                else None
            ),
            "best_downstream_gap_to_teacher_importance_topk_mse": (
                float(best_mse["student_future_mse"]) - float(teacher_topk["student_future_mse"])
                if best_mse is not None and _finite(teacher_topk.get("student_future_mse"))
                else None
            ),
        }
    return {
        "num_variants": len(rows),
        "num_successful": len(successful),
        "failed_variants": [row for row in rows if not row.get("success", True)],
        "best_by_student_future_mse": best_mse,
        "step12_reference": step12_refs,
        "comparison": comparison,
        "sanity_gate": {"checks": checks, "pass": all(checks.values())},
    }


def render_selector_markdown(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    lines = [
        "| variant | final_loss | importance_mse | pearson_corr | top1_overlap | topk_overlap | selected_teacher_importance_mean | selected_vs_random_importance_gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: str(item.get("variant_run_name", item.get("variant_name", "")))):
        lines.append(
            "| {variant} | {final_loss} | {mse} | {corr} | {top1} | {topk} | {importance} | {gap} |".format(
                variant=row.get("variant_name"),
                final_loss=_fmt(row.get("final_loss")),
                mse=_fmt(row.get("test_importance_mse")),
                corr=_fmt(row.get("test_pearson_corr_mean"), precision=6),
                top1=_fmt(row.get("test_target_top1_overlap"), precision=6),
                topk=_fmt(row.get("test_target_topk_overlap"), precision=6),
                importance=_fmt(row.get("test_selected_teacher_importance_mean"), precision=6),
                gap=_fmt(row.get("test_selected_vs_random_importance_gap"), precision=6),
            )
        )
    lines.extend(["", "## Selector Aggregate", "", "```json", json.dumps(aggregate, indent=2), "```"])
    return "\n".join(lines) + "\n"


def render_downstream_markdown(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    lines = [
        "| variant | student_future_mse | teacher_mse | student_teacher_ratio | token_retention_ratio | selector_target_topk_overlap | selected_teacher_importance_mean | selected_vs_random_importance_gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: str(item.get("variant_run_name", item.get("variant_name", "")))):
        lines.append(
            "| {variant} | {mse} | {teacher} | {ratio} | {retention} | {topk} | {importance} | {gap} |".format(
                variant=row.get("variant_name"),
                mse=_fmt(row.get("student_future_mse")),
                teacher=_fmt(row.get("teacher_mse", row.get("teacher_future_mse"))),
                ratio=_fmt(row.get("student_teacher_ratio"), precision=6),
                retention=_fmt(row.get("token_retention_ratio"), precision=6),
                topk=_fmt(row.get("selector_target_topk_overlap"), precision=6),
                importance=_fmt(row.get("selected_teacher_importance_mean"), precision=6),
                gap=_fmt(row.get("selected_vs_random_importance_gap"), precision=6),
            )
        )
    lines.extend(["", "## Downstream Aggregate", "", "```json", json.dumps(aggregate, indent=2), "```"])
    return "\n".join(lines) + "\n"


def render_step12_comparison_markdown(step12_reference: dict[str, Any]) -> str:
    rows = []
    random_mean = step12_reference.get("random_k_mean")
    if random_mean:
        rows.append(("step12_random_k_mean", random_mean, "mean over seeds 0,1,2"))
    for key, note in (
        ("uniform_k", "seed0"),
        ("teacher_importance_topk", "oracle-like upper bound seed0"),
        ("learned_selector", "Step 12 learned selector seed0"),
    ):
        if step12_reference.get(key):
            rows.append((f"step12_{key}", step12_reference[key], note))
    lines = [
        "| method | student_future_mse | selected_teacher_importance_mean | topk_overlap | note |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for method, row, note in rows:
        lines.append(
            "| {method} | {mse} | {importance} | {topk} | {note} |".format(
                method=method,
                mse=_fmt(row.get("student_future_mse")),
                importance=_fmt(row.get("selected_teacher_importance_mean"), precision=6),
                topk=_fmt(row.get("selector_target_topk_overlap"), precision=6),
                note=note,
            )
        )
    return "\n".join(lines) + "\n"


def write_selector_ablation_summaries(rows: list[dict[str, Any]], run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    aggregate = aggregate_selector_rows(rows)
    json_path = run_path / "selector_ablation_summary.json"
    csv_path = run_path / "selector_ablation_summary.csv"
    md_path = run_path / "selector_ablation_summary.md"
    payload = {"run_dir": str(run_path), "rows": rows, "aggregate": aggregate}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTOR_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in SELECTOR_SUMMARY_COLUMNS})
    md_path.write_text(render_selector_markdown(rows, aggregate), encoding="utf-8")
    return {
        "selector_summary_json": str(json_path),
        "selector_summary_csv": str(csv_path),
        "selector_summary_md": str(md_path),
        "selector_aggregate": aggregate,
    }


def write_downstream_ablation_summaries(
    rows: list[dict[str, Any]],
    run_dir: str | Path,
    step12_summary_json: str | Path | None = None,
    selector_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    aggregate = aggregate_downstream_rows(rows, step12_summary_json=step12_summary_json)
    json_path = run_path / "downstream_ablation_summary.json"
    csv_path = run_path / "downstream_ablation_summary.csv"
    md_path = run_path / "downstream_ablation_summary.md"
    combined_json_path = run_path / "selector_ablation_combined_report.json"
    combined_md_path = run_path / "selector_ablation_combined_report.md"
    payload = {"run_dir": str(run_path), "rows": rows, "aggregate": aggregate}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DOWNSTREAM_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in DOWNSTREAM_SUMMARY_COLUMNS})
    md_path.write_text(render_downstream_markdown(rows, aggregate), encoding="utf-8")

    selector_payload = selector_summary or {}
    comparison_md = render_step12_comparison_markdown(aggregate.get("step12_reference", {}))
    combined = {
        "run_dir": str(run_path),
        "selector_summary": selector_payload,
        "downstream_summary": payload,
        "comparison": aggregate.get("comparison", {}),
        "sanity_gate": {
            "selector_pass": bool(selector_payload.get("aggregate", {}).get("sanity_gate", {}).get("pass", False)),
            "downstream_pass": bool(aggregate.get("sanity_gate", {}).get("pass", False)),
            "pass": bool(selector_payload.get("aggregate", {}).get("sanity_gate", {}).get("pass", False))
            and bool(aggregate.get("sanity_gate", {}).get("pass", False)),
        },
    }
    combined_json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    combined_md_path.write_text(
        "# BAIR Selector Ablation Combined Report\n\n"
        "## Selector-Level Results\n\n"
        + Path(run_path / "selector_ablation_summary.md").read_text(encoding="utf-8")
        + "\n## Downstream StudentWorldModel Results\n\n"
        + Path(md_path).read_text(encoding="utf-8")
        + "\n## Comparison With Step 12\n\n"
        + comparison_md
        + "\n## Combined Sanity Gate\n\n```json\n"
        + json.dumps(combined["sanity_gate"], indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return {
        "downstream_summary_json": str(json_path),
        "downstream_summary_csv": str(csv_path),
        "downstream_summary_md": str(md_path),
        "combined_report_json": str(combined_json_path),
        "combined_report_md": str(combined_md_path),
        "downstream_aggregate": aggregate,
        "combined_report": combined,
    }


def run_selector_ablation(config: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    selector_rows = train_selector_variants(config, run_dir=run_dir)
    selector_paths = write_selector_ablation_summaries(selector_rows, run_dir=run_dir)
    downstream_rows = train_downstream_variants(config, selector_rows, run_dir=run_dir)
    downstream_paths = write_downstream_ablation_summaries(
        downstream_rows,
        run_dir=run_dir,
        step12_summary_json=config.get("step12_reference", {}).get("baseline_summary_json"),
        selector_summary={
            "rows": selector_rows,
            "aggregate": selector_paths["selector_aggregate"],
        },
    )
    result = {
        "run_dir": str(run_dir),
        "loss_variants": [variant["name"] for variant in config.get("loss_variants", [])],
        "selector_rows": selector_rows,
        "downstream_rows": downstream_rows,
        **selector_paths,
        **downstream_paths,
    }
    print(json.dumps(result, indent=2))
    return result
