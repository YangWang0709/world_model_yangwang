"""Step18 BAIR context-selector oracle-gap training and diagnostics."""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.context_token_shard_dataset import ContextSelectorDataset, context_selector_collate_fn, summarize_context_dataset
from eval.eval_context_importance_diagnostic import write_context_importance_diagnostic
from models.context_bottleneck_world_model import ContextBottleneckWorldModel, future_target_from_tokens
from models.context_selection_policies import topk_from_retention
from models.context_selector_losses import compute_context_selector_loss
from models.selection_policies import compute_teacher_importance_selection_metrics, gather_tokens_by_indices
from models.unified_predictive_importance_selector import (
    UnifiedPredictiveImportanceSelector,
    build_unified_selector_from_config,
    initialize_context_selector_from_state_checkpoint,
)
from scripts.run_bair_500_64_scale_validation import ensure_resource_limits, resource_snapshot
from training.losses import future_latent_mse, importance_topk_metrics
from training.teacher_trainer import grad_norm, resolve_device
from training.train_unified_context_selector import save_unified_selector_checkpoint


STAGE = "context_selector_oracle_gap_bair_1000_128"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "context_selector_oracle_gap_bair_1000_128.yaml"
PROTECTED_OUTPUT_MARKERS = (
    "bair_context_windows_1000_128",
    "context_token_shards",
    "context_importance_shards",
    "context_teacher_bair_1000_128",
    "unified_context_selector_bair_1000_128",
    "context_bottleneck_world_model_bair_1000_128",
    "context_bottleneck_baseline_bair_1000_128",
    "context_bottleneck_bair_1000_128",
    "student_selector_bair_videomae_1000_128_weighted_mse_multiseed_v1",
    "downstream_utilization_bair_1000_128_v1",
)


@dataclass(frozen=True)
class StepDefinition:
    name: str
    description: str


STEP_DEFINITIONS = [
    StepDefinition("resource_check", "check disk/RAM/GPU and fixed Step17 inputs"),
    StepDefinition("validate_step17_inputs", "verify Step17 context tokens, importance, teacher, and baselines"),
    StepDefinition("label_diagnostic", "diagnose context-importance label quality"),
    StepDefinition("phase_a_selectors", "train and evaluate seed0 selector variants"),
    StepDefinition("phase_a_downstream", "train downstream models for top Phase A variants"),
    StepDefinition("phase_b_selectors", "retrain top variants across seeds"),
    StepDefinition("phase_b_downstream", "train downstream models for Phase B seeds"),
    StepDefinition("write_summary", "write oracle-gap summaries"),
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return payload


def step_names() -> list[str]:
    return [step.name for step in STEP_DEFINITIONS]


def variant_run_name(variant: str, seed: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(variant))
    return f"{safe}_seed{int(seed)}"


def is_step18_owned_output_path(path_value: str | Path) -> bool:
    value = str(path_value)
    if any(marker in value for marker in PROTECTED_OUTPUT_MARKERS):
        return False
    return "context_selector_oracle_gap_bair_1000_128" in value


def write_partial_summary(
    run_dir: str | Path,
    *,
    current_step: str,
    completed_steps: list[str],
    status: str,
    error: str | None = None,
    resource: dict[str, Any] | None = None,
) -> Path:
    path = Path(run_dir) / "partial_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": STAGE,
        "status": status,
        "current_step": current_step,
        "completed_steps": completed_steps,
        "error": error,
        "resource": resource or {},
        "updated_at_unix": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _resource_delta(before: dict[str, Any], after: dict[str, Any], elapsed: float, oom: bool) -> dict[str, Any]:
    gpu_total = after.get("gpu_mem_total_gib") or 0.0
    gpu_used = after.get("gpu_mem_used_gib") or 0.0
    return {
        "before": before,
        "after": after,
        "elapsed_time_sec": round(float(elapsed), 3),
        "max_ram_used_gib": after.get("ram_used_gib"),
        "max_gpu_mem_used_gib": after.get("gpu_mem_used_gib"),
        "gpu_mem_fraction": float(gpu_used) / float(gpu_total) if gpu_total else None,
        "oom": bool(oom),
        "cloud_recommendation": "recommended only after lowering batch_size still fails" if oom else "not required for Step18 local oracle-gap diagnostic",
    }


def _read_json(path: str | Path, required: bool = True) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(p)
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {p}")
    return payload


def _write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def _table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        rendered = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                rendered.append(f"{value:.8f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _variant_loss_config(variant: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(variant)
    if cfg.get("loss_type") == "temporal_block_balanced_topk":
        cfg["loss_type"] = "temporal_block_balanced_topk"
    return cfg


def _temporal_block_ids(num_tokens: int, temporal_blocks: int, device: torch.device) -> torch.Tensor:
    return torch.arange(num_tokens, device=device).mul(int(temporal_blocks)).floor_divide(int(num_tokens)).clamp_max(temporal_blocks - 1)


def _block_balanced_indices(scores: torch.Tensor, *, temporal_blocks: int, topk_per_block: int, total_topk: int) -> torch.Tensor:
    block_ids = _temporal_block_ids(scores.shape[1], temporal_blocks, scores.device)
    rows = []
    for sample_scores in scores:
        selected = []
        for block_id in range(int(temporal_blocks)):
            block_indices = torch.nonzero(block_ids == block_id, as_tuple=False).squeeze(1)
            if block_indices.numel() == 0:
                continue
            k = min(int(topk_per_block), int(block_indices.numel()))
            selected.append(block_indices[torch.topk(sample_scores[block_indices], k=k).indices])
        merged = torch.cat(selected, dim=0) if selected else torch.empty(0, dtype=torch.long, device=scores.device)
        if merged.numel() < total_topk:
            fill = torch.topk(sample_scores, k=total_topk).indices
            merged = torch.unique(torch.cat([merged, fill], dim=0), sorted=False)
        if merged.numel() > total_topk:
            merged = merged[torch.topk(sample_scores[merged], k=total_topk).indices]
        rows.append(merged[:total_topk])
    return torch.stack(rows, dim=0)


def select_indices_from_scores(scores: torch.Tensor, variant: dict[str, Any], topk: int) -> torch.Tensor:
    if str(variant.get("name")) == "temporal_block_balanced_topk":
        return _block_balanced_indices(
            scores,
            temporal_blocks=int(variant.get("temporal_blocks", 8)),
            topk_per_block=int(variant.get("topk_per_block", max(1, topk // int(variant.get("temporal_blocks", 8))))),
            total_topk=topk,
        )
    return torch.topk(scores, k=topk, dim=1).indices


def temporal_block_coverage(selected_indices: torch.Tensor, *, num_tokens: int, temporal_blocks: int = 8) -> float:
    if selected_indices.numel() == 0:
        return 0.0
    block_ids = torch.arange(num_tokens).mul(int(temporal_blocks)).floor_divide(int(num_tokens)).clamp_max(temporal_blocks - 1)
    coverages = []
    for row in selected_indices.cpu().long():
        selected_blocks = set(int(block_ids[index].item()) for index in row)
        coverages.append(len(selected_blocks) / float(temporal_blocks))
    return float(sum(coverages) / len(coverages)) if coverages else 0.0


def _selector_metrics_from_scores(scores: torch.Tensor, target: torch.Tensor, selected_indices: torch.Tensor, topk: int, seed: int) -> dict[str, float]:
    score_metrics = importance_topk_metrics(scores, target, k=topk, random_seed=seed)
    selection_metrics = compute_teacher_importance_selection_metrics(
        selected_indices.cpu(),
        target.cpu(),
        k=topk,
        num_tokens=int(target.shape[1]),
        random_seed=seed,
    )
    return {
        "context_importance_mse": score_metrics["importance_mse"],
        "context_importance_mae": score_metrics["importance_mae"],
        "pearson_corr_mean": score_metrics["pearson_corr_mean"],
        "target_top1_overlap": selection_metrics["selector_target_top1_overlap"],
        "target_topk_overlap": selection_metrics["selector_target_topk_overlap"],
        "selected_context_importance_mean": selection_metrics["selected_teacher_importance_mean"],
        "random_context_importance_mean": selection_metrics["random_teacher_importance_mean"],
        "selected_vs_random_context_importance_gap": selection_metrics["selected_vs_random_importance_gap"],
        "temporal_block_coverage": temporal_block_coverage(selected_indices, num_tokens=int(target.shape[1])),
    }


def _dataset(config: dict[str, Any], split: str) -> ContextSelectorDataset:
    data_cfg = config["data"]
    max_samples = data_cfg.get(f"max_{split}_samples")
    return ContextSelectorDataset(
        token_shard_dir=data_cfg[f"context_token_shard_dir_{split}"],
        importance_shard_dir=data_cfg[f"context_importance_shard_dir_{split}"],
        token_shard_glob=data_cfg.get("token_shard_glob", "context_tokens_shard_*.pt"),
        importance_shard_glob=data_cfg.get("importance_shard_glob", "importance_shard_*.pt"),
        max_samples=int(max_samples) if max_samples is not None else None,
    )


class ContextSelectorOracleGapRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
        if not is_step18_owned_output_path(self.run_dir):
            raise ValueError(f"Refusing non-Step18 run dir: {self.run_dir}")
        self.device = resolve_device(str(config["selector_training"].get("device", "cuda_if_available")))
        self.topk = int(config["selection"]["context_topk"])
        self.train_dataset: ContextSelectorDataset | None = None
        self.test_dataset: ContextSelectorDataset | None = None
        self.train_loader: DataLoader | None = None
        self.eval_loader: DataLoader | None = None

    def validate_inputs(self) -> None:
        paths = [
            self.config["data"]["context_token_shard_dir_train"],
            self.config["data"]["context_token_shard_dir_test"],
            self.config["data"]["context_importance_shard_dir_train"],
            self.config["data"]["context_importance_shard_dir_test"],
            self.config["teacher_reference"]["checkpoint"],
            self.config["step17_reference"]["context_bottleneck_summary"],
            self.config["step17_reference"]["baseline_aggregate"],
        ]
        missing = [str(path) for path in paths if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Missing Step17 input(s): {missing}")
        if bool(self.config["selection"].get("current_tokens_dropped", True)):
            raise ValueError("Step18 requires current_tokens_dropped=false")
        if bool(self.config["selection"].get("train_current_importance", True)):
            raise ValueError("Step18 must not train current importance")

    def _load_data(self) -> None:
        if self.train_dataset is not None:
            return
        batch_size = int(self.config["selector_training"]["batch_size"])
        num_workers = int(self.config["selector_training"].get("num_workers", 0))
        self.train_dataset = _dataset(self.config, "train")
        self.test_dataset = _dataset(self.config, "test")
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=context_selector_collate_fn)
        self.eval_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=context_selector_collate_fn)
        sample = self.train_dataset[0]
        expected_context = int(self.config["data"]["expected_context_tokens"])
        expected_current = int(self.config["data"]["expected_current_tokens"])
        expected_future = int(self.config["data"]["expected_future_tokens"])
        if int(sample["context_tokens"].shape[0]) != expected_context:
            raise ValueError("Unexpected context token count")
        if int(sample["current_tokens"].shape[0]) != expected_current:
            raise ValueError("Unexpected current token count")
        if int(sample["future_tokens"].shape[0]) != expected_future:
            raise ValueError("Unexpected future token count")
        computed_topk = topk_from_retention(
            expected_context,
            float(self.config["selection"]["retention_ratio"]),
            int(self.config["selection"]["context_topk"]),
        )
        if computed_topk != self.topk:
            raise ValueError(f"context_topk mismatch: config={self.topk}, retention gives {computed_topk}")

    def _build_selector(self, variant: dict[str, Any]) -> tuple[UnifiedPredictiveImportanceSelector, bool, dict[str, Any]]:
        model_cfg = dict(self.config["selector_model"])
        model_cfg["condition_on_current"] = bool(variant.get("condition_on_current", model_cfg.get("condition_on_current", True)))
        model_cfg["temporal_position_embedding"] = bool(variant.get("temporal_position_embedding", model_cfg.get("temporal_position_embedding", True)))
        model_cfg.pop("mode", None)
        model_cfg.pop("current_pool", None)
        model = build_unified_selector_from_config(model_cfg).to(self.device)
        init_cfg = self.config.get("selector_model", {})
        initialized = False
        if bool(init_cfg.get("init_from_step15_state_selector", False)) and Path(str(init_cfg.get("state_selector_checkpoint", ""))).exists():
            initialized = initialize_context_selector_from_state_checkpoint(
                model,
                init_cfg["state_selector_checkpoint"],
                allow_random_init_if_incompatible=bool(init_cfg.get("allow_random_init_if_incompatible", True)),
            )
        return model, initialized, model_cfg

    def evaluate_selector(self, model: UnifiedPredictiveImportanceSelector, variant: dict[str, Any], seed: int) -> dict[str, float]:
        assert self.eval_loader is not None
        model.eval()
        all_scores = []
        all_targets = []
        all_indices = []
        with torch.no_grad():
            for batch in self.eval_loader:
                context = batch["context_tokens"].to(self.device)
                current = batch["current_tokens"].to(self.device)
                logits = model(context_tokens=context, current_tokens=current, mode="context")
                scores = torch.sigmoid(logits)
                indices = select_indices_from_scores(scores, variant, self.topk)
                all_scores.append(scores.cpu())
                all_targets.append(batch["importance_scores_norm"].float().cpu())
                all_indices.append(indices.cpu())
        score = torch.cat(all_scores, dim=0)
        target = torch.cat(all_targets, dim=0)
        indices = torch.cat(all_indices, dim=0)
        return _selector_metrics_from_scores(score, target, indices, self.topk, seed)

    def train_selector_variant(self, variant: dict[str, Any], *, seed: int, phase: str) -> dict[str, Any]:
        self._load_data()
        assert self.train_loader is not None
        _seed_everything(seed)
        model, initialized, model_cfg = self._build_selector(variant)
        train_cfg = self.config["selector_training"]
        max_steps = int(variant.get("max_steps", train_cfg.get("max_steps", 800)))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        run_dir = self.run_dir / f"{phase}_selectors" / variant_run_name(str(variant["name"]), seed)
        checkpoint_dir = run_dir / "checkpoints"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.jsonl"
        metrics_path.write_text("", encoding="utf-8")
        temporal_ids = _temporal_block_ids(int(self.config["data"]["expected_context_tokens"]), int(variant.get("temporal_blocks", 8)), self.device)
        rows: list[dict[str, Any]] = []
        step = 0
        while step < max_steps:
            model.train()
            for batch in self.train_loader:
                step += 1
                context = batch["context_tokens"].to(self.device)
                current = batch["current_tokens"].to(self.device)
                target = batch["importance_scores_norm"].to(self.device)
                logits = model(context_tokens=context, current_tokens=current, mode="context")
                loss_parts = compute_context_selector_loss(logits, target, _variant_loss_config(variant), temporal_block_ids=temporal_ids)
                loss = loss_parts["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = grad_norm(model.parameters())
                max_grad = float(train_cfg.get("max_grad_norm", 0.0))
                if max_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                optimizer.step()
                row = {
                    "step": int(step),
                    "loss": float(loss.item()),
                    "mse_loss": float(loss_parts["mse_loss"].item()),
                    "weighted_mse_loss": float(loss_parts["weighted_mse_loss"].item()),
                    "bce_loss": float(loss_parts["bce_loss"].item()),
                    "rank_loss": float(loss_parts["rank_loss"].item()),
                    "temporal_balance_loss": float(loss_parts["temporal_balance_loss"].item()),
                    "grad_norm": float(norm),
                }
                rows.append(row)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                if step >= max_steps:
                    break
        losses = [row["loss"] for row in rows]
        eval_metrics = self.evaluate_selector(model, variant, seed)
        summary: dict[str, Any] = {
            "variant": str(variant["name"]),
            "seed": int(seed),
            "phase": phase,
            "loss_type": str(variant["loss_type"]),
            "condition_on_current": bool(variant.get("condition_on_current", True)),
            "temporal_position_embedding": bool(variant.get("temporal_position_embedding", True)),
            "num_steps": int(len(rows)),
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "best_loss": float(min(losses)),
            "loss_decreased": bool(losses[-1] < losses[0]),
            "context_topk": int(self.topk),
            "context_retention_ratio": float(self.topk) / float(self.config["data"]["expected_context_tokens"]),
            "current_tokens_dropped": False,
            "trained_current_importance": False,
            "trained_context_importance": True,
            "initialized_from_step15_selector": bool(initialized),
            "metrics_path": str(metrics_path),
            "run_dir": str(run_dir),
            **eval_metrics,
        }
        checkpoint_path = checkpoint_dir / f"unified_selector_context_step_{len(rows):06d}.pt"
        summary["checkpoint_path"] = str(save_unified_selector_checkpoint(checkpoint_path, model, optimizer, len(rows), model_cfg, summary))
        _write_json(run_dir / "summary.json", summary)
        return summary

    def train_downstream_for_selector(self, selector_summary: dict[str, Any], variant: dict[str, Any], *, seed: int, phase: str) -> dict[str, Any]:
        self._load_data()
        assert self.train_loader is not None and self.eval_loader is not None
        _seed_everything(seed)
        downstream_cfg = self.config["downstream_eval"]
        model_cfg = dict(self.config.get("downstream_model", {}))
        model = ContextBottleneckWorldModel(**model_cfg).to(self.device)
        selector_ckpt = torch.load(selector_summary["checkpoint_path"], map_location="cpu")
        selector = build_unified_selector_from_config(selector_ckpt["model_config"])
        selector.load_state_dict(selector_ckpt["model_state_dict"])
        selector.to(self.device)
        selector.eval()
        for parameter in selector.parameters():
            parameter.requires_grad = False
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(downstream_cfg.get("learning_rate", 0.001)),
            weight_decay=float(downstream_cfg.get("weight_decay", 0.0001)),
        )
        max_steps = int(variant.get("downstream_max_steps", downstream_cfg.get("max_steps", 800)))
        run_dir = self.run_dir / f"{phase}_downstream" / variant_run_name(str(selector_summary["variant"]), seed)
        checkpoint_dir = run_dir / "checkpoints"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.jsonl"
        metrics_path.write_text("", encoding="utf-8")
        rows: list[dict[str, Any]] = []
        step = 0
        while step < max_steps:
            model.train()
            for batch in self.train_loader:
                step += 1
                context = batch["context_tokens"].to(self.device)
                current = batch["current_tokens"].to(self.device)
                future = batch["future_tokens"].to(self.device)
                target = future_target_from_tokens(future)
                with torch.no_grad():
                    scores = torch.sigmoid(selector(context_tokens=context, current_tokens=current, mode="context"))
                    indices = select_indices_from_scores(scores, variant, self.topk)
                    selected = gather_tokens_by_indices(context, indices)
                    selected_scores = torch.gather(scores, dim=1, index=indices)
                pred = model(current, selected, selected_scores if model.context_memory_mode == "selector_score_weighted_pool" else None)
                loss = future_latent_mse(pred, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = grad_norm(model.parameters())
                if float(downstream_cfg.get("max_grad_norm", 1.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(downstream_cfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                row = {"step": int(step), "loss": float(loss.item()), "grad_norm": float(norm)}
                rows.append(row)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                if step >= max_steps:
                    break
        eval_metrics = self.evaluate_downstream(model, selector, variant, seed)
        refs = self.config["step17_reference"]
        future_mse = float(eval_metrics["future_mse"])
        oracle_mse = float(refs["teacher_context_importance_topk_mse"])
        summary = {
            "variant": str(selector_summary["variant"]),
            "seed": int(seed),
            "phase": phase,
            "future_mse": future_mse,
            "student_future_mse": future_mse,
            "current_only_mse_reference": float(refs["current_only_mse"]),
            "random_context_topk_mse_reference": float(refs["random_context_topk_mse"]),
            "learned_context_step17_mse_reference": float(refs["learned_context_selector_topk_mse"]),
            "teacher_context_importance_topk_mse_reference": oracle_mse,
            "beats_step17_learned": bool(future_mse < float(refs["learned_context_selector_topk_mse"])),
            "beats_random_context": bool(future_mse < float(refs["random_context_topk_mse"])),
            "beats_current_only": bool(future_mse < float(refs["current_only_mse"])),
            "oracle_gap": float(future_mse - oracle_mse),
            "num_steps": int(len(rows)),
            "initial_loss": float(rows[0]["loss"]),
            "final_loss": float(rows[-1]["loss"]),
            "best_loss": float(min(row["loss"] for row in rows)),
            "loss_decreased": bool(rows[-1]["loss"] < rows[0]["loss"]),
            "current_tokens_dropped": False,
            "trained_current_importance": False,
            "trained_context_importance": True,
            "selector_checkpoint_path": selector_summary["checkpoint_path"],
            "metrics_path": str(metrics_path),
            "run_dir": str(run_dir),
            **eval_metrics,
        }
        ckpt_path = checkpoint_dir / f"context_bottleneck_world_model_step_{len(rows):06d}.pt"
        torch.save(
            {
                "step": int(len(rows)),
                "model_config": model_cfg,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics_summary": summary,
                "current_tokens_dropped": False,
            },
            ckpt_path,
        )
        summary["checkpoint_path"] = str(ckpt_path)
        _write_json(run_dir / "summary.json", summary)
        return summary

    def evaluate_downstream(
        self,
        model: ContextBottleneckWorldModel,
        selector: UnifiedPredictiveImportanceSelector,
        variant: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        assert self.eval_loader is not None
        model.eval()
        selector.eval()
        sse = 0.0
        count = 0
        selected_indices_all = []
        target_importance_all = []
        with torch.no_grad():
            for batch in self.eval_loader:
                context = batch["context_tokens"].to(self.device)
                current = batch["current_tokens"].to(self.device)
                future = batch["future_tokens"].to(self.device)
                target = future_target_from_tokens(future)
                scores = torch.sigmoid(selector(context_tokens=context, current_tokens=current, mode="context"))
                indices = select_indices_from_scores(scores, variant, self.topk)
                selected = gather_tokens_by_indices(context, indices)
                selected_scores = torch.gather(scores, dim=1, index=indices)
                pred = model(current, selected, selected_scores if model.context_memory_mode == "selector_score_weighted_pool" else None)
                sse += float((pred - target).pow(2).sum().item())
                count += int(target.numel())
                selected_indices_all.append(indices.cpu())
                target_importance_all.append(batch["importance_scores_norm"].float().cpu())
        mse = sse / float(max(count, 1))
        indices = torch.cat(selected_indices_all, dim=0)
        target_importance = torch.cat(target_importance_all, dim=0)
        sel_metrics = compute_teacher_importance_selection_metrics(
            indices,
            target_importance,
            k=self.topk,
            num_tokens=int(target_importance.shape[1]),
            random_seed=seed,
        )
        return {
            "future_mse": float(mse),
            "selected_context_importance_mean": sel_metrics["selected_teacher_importance_mean"],
            "random_context_importance_mean": sel_metrics["random_teacher_importance_mean"],
            "selected_vs_random_context_importance_gap": sel_metrics["selected_vs_random_importance_gap"],
            "target_topk_overlap": sel_metrics["selector_target_topk_overlap"],
            "temporal_block_coverage": temporal_block_coverage(indices, num_tokens=int(target_importance.shape[1])),
        }

    def _write_selector_summary(self, name: str, rows: list[dict[str, Any]]) -> None:
        _write_json(self.run_dir / f"{name}.json", {"rows": rows})
        columns = [
            "variant",
            "seed",
            "loss_type",
            "final_loss",
            "context_importance_mse",
            "pearson_corr_mean",
            "target_topk_overlap",
            "selected_context_importance_mean",
            "temporal_block_coverage",
        ]
        _write_csv(self.run_dir / f"{name}.csv", rows, columns)
        (self.run_dir / f"{name}.md").write_text(_table_md(rows, columns) + "\n", encoding="utf-8")

    def _write_downstream_summary(self, name: str, rows: list[dict[str, Any]]) -> None:
        _write_json(self.run_dir / f"{name}.json", {"rows": rows})
        columns = [
            "variant",
            "seed",
            "future_mse",
            "beats_step17_learned",
            "beats_random_context",
            "beats_current_only",
            "oracle_gap",
            "selected_context_importance_mean",
            "target_topk_overlap",
            "temporal_block_coverage",
        ]
        _write_csv(self.run_dir / f"{name}.csv", rows, columns)
        (self.run_dir / f"{name}.md").write_text(_table_md(rows, columns) + "\n", encoding="utf-8")

    def _aggregate_phase_b_selectors(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["variant"]), []).append(row)
        aggregate = []
        for variant, items in sorted(grouped.items()):
            topk = torch.tensor([float(item["target_topk_overlap"]) for item in items])
            importance = torch.tensor([float(item["selected_context_importance_mean"]) for item in items])
            aggregate.append(
                {
                    "variant": variant,
                    "seeds": ",".join(str(item["seed"]) for item in items),
                    "n": len(items),
                    "topk_overlap_mean": float(topk.mean().item()),
                    "topk_overlap_std": float(topk.std(unbiased=False).item()) if len(items) > 1 else 0.0,
                    "selected_importance_mean": float(importance.mean().item()),
                    "selected_importance_std": float(importance.std(unbiased=False).item()) if len(items) > 1 else 0.0,
                }
            )
        return {"rows": rows, "aggregate": aggregate}

    def _aggregate_phase_b_downstream(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["variant"]), []).append(row)
        aggregate = []
        for variant, items in sorted(grouped.items()):
            mse = torch.tensor([float(item["future_mse"]) for item in items])
            gap = torch.tensor([float(item["oracle_gap"]) for item in items])
            aggregate.append(
                {
                    "variant": variant,
                    "seeds": ",".join(str(item["seed"]) for item in items),
                    "n": len(items),
                    "future_mse_mean": float(mse.mean().item()),
                    "future_mse_std": float(mse.std(unbiased=False).item()) if len(items) > 1 else 0.0,
                    "oracle_gap_mean": float(gap.mean().item()),
                    "oracle_gap_std": float(gap.std(unbiased=False).item()) if len(items) > 1 else 0.0,
                }
            )
        return {"rows": rows, "aggregate": aggregate}

    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        completed: list[str] = []
        before = resource_snapshot()
        ensure_resource_limits(self.config, before)
        start = time.perf_counter()
        oom = False
        try:
            completed.append("resource_check")
            write_partial_summary(self.run_dir, current_step="resource_check", completed_steps=completed, status="running", resource={"before": before})
            self.validate_inputs()
            completed.append("validate_step17_inputs")
            label_summary = write_context_importance_diagnostic(self.config)
            completed.append("label_diagnostic")
            self._load_data()
            phase_a_selectors = []
            for variant in self.config["selector_variants"]:
                phase_a_selectors.append(self.train_selector_variant(variant, seed=int(variant.get("seed", 0)), phase="phase_a"))
            self._write_selector_summary("selector_phase_a_summary", phase_a_selectors)
            completed.append("phase_a_selectors")
            ranked_for_downstream = sorted(
                phase_a_selectors,
                key=lambda row: (float(row["target_topk_overlap"]), float(row["selected_context_importance_mean"])),
                reverse=True,
            )
            top_n = int(self.config["downstream_eval"].get("top_n_variants", 4))
            variant_by_name = {str(item["name"]): dict(item) for item in self.config["selector_variants"]}
            phase_a_downstream = []
            for selector_summary in ranked_for_downstream[:top_n]:
                variant = variant_by_name[str(selector_summary["variant"])]
                phase_a_downstream.append(self.train_downstream_for_selector(selector_summary, variant, seed=int(selector_summary["seed"]), phase="phase_a"))
            self._write_downstream_summary("downstream_phase_a_summary", phase_a_downstream)
            completed.append("phase_a_downstream")
            if phase_a_downstream:
                ranked_for_phase_b = sorted(phase_a_downstream, key=lambda row: float(row["future_mse"]))
                phase_b_names = [str(row["variant"]) for row in ranked_for_phase_b[: int(self.config["phase_b"].get("select_top_n_from_phase_a", 2))]]
            else:
                phase_b_names = [str(row["variant"]) for row in ranked_for_downstream[:2]]
            phase_b_selector_rows = []
            phase_b_downstream_rows = []
            for name in phase_b_names:
                variant = variant_by_name[name]
                for seed in [int(seed) for seed in self.config["phase_b"].get("seeds", [0, 1, 2])]:
                    variant_b = dict(variant)
                    variant_b["max_steps"] = int(min(int(self.config["phase_b"].get("selector_max_steps", 800)), int(self.config["selector_training"].get("max_steps", 800))))
                    variant_b["downstream_max_steps"] = int(min(int(self.config["phase_b"].get("downstream_max_steps", 800)), int(self.config["downstream_eval"].get("max_steps", 800))))
                    selector_summary = self.train_selector_variant(variant_b, seed=seed, phase="phase_b")
                    phase_b_selector_rows.append(selector_summary)
                    phase_b_downstream_rows.append(self.train_downstream_for_selector(selector_summary, variant_b, seed=seed, phase="phase_b"))
            self._write_selector_summary("selector_phase_b_summary", phase_b_selector_rows)
            _write_json(self.run_dir / "selector_phase_b_aggregate.json", self._aggregate_phase_b_selectors(phase_b_selector_rows))
            completed.append("phase_b_selectors")
            self._write_downstream_summary("downstream_phase_b_summary", phase_b_downstream_rows)
            _write_json(self.run_dir / "downstream_phase_b_aggregate.json", self._aggregate_phase_b_downstream(phase_b_downstream_rows))
            completed.append("phase_b_downstream")
            after = resource_snapshot()
            resource = _resource_delta(before, after, time.perf_counter() - start, oom=False)
            summary = {
                "stage": STAGE,
                "run_dir": str(self.run_dir),
                "label_diagnostic": label_summary,
                "selector_phase_a_rows": phase_a_selectors,
                "downstream_phase_a_rows": phase_a_downstream,
                "selector_phase_b_rows": phase_b_selector_rows,
                "downstream_phase_b_rows": phase_b_downstream_rows,
                "selector_phase_b_aggregate": self._aggregate_phase_b_selectors(phase_b_selector_rows),
                "downstream_phase_b_aggregate": self._aggregate_phase_b_downstream(phase_b_downstream_rows),
                "resource_summary": resource,
                "current_tokens_dropped": False,
                "trained_current_importance": False,
                "trained_context_importance": True,
            }
            _write_json(self.run_dir / "runner_summary.json", summary)
            completed.append("write_summary")
            write_partial_summary(self.run_dir, current_step="complete", completed_steps=completed, status="complete", resource=resource)
            return summary
        except BaseException as exc:
            oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower() or "oom" in str(exc).lower()
            if oom and torch.cuda.is_available():
                torch.cuda.empty_cache()
            after = resource_snapshot()
            resource = _resource_delta(before, after, time.perf_counter() - start, oom=oom)
            write_partial_summary(self.run_dir, current_step=completed[-1] if completed else "resource_check", completed_steps=completed, status="failed", error=repr(exc), resource=resource)
            raise


def run_bair_context_selector_oracle_gap(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return ContextSelectorOracleGapRunner(load_yaml(config_path)).run()
