"""Generate context-only Teacher occlusion importance for Step 17."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.context_token_shard_dataset import ContextTokenShardDataset, context_token_collate_fn, summarize_context_dataset
from data.importance_shards import save_importance_shard, summarize_importance_shard, utc_now_iso
from models.context_bottleneck_world_model import ContextTeacherWorldModel, future_target_from_tokens
from training.teacher_trainer import resolve_device


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _load_teacher(path: str | Path, device: torch.device) -> tuple[ContextTeacherWorldModel, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    model = ContextTeacherWorldModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, checkpoint


def _normalize(scores: torch.Tensor) -> torch.Tensor:
    mins = scores.min(dim=1, keepdim=True).values
    maxs = scores.max(dim=1, keepdim=True).values
    denom = (maxs - mins).clamp_min(torch.finfo(scores.dtype).eps)
    norm = (scores - mins) / denom
    constant = (maxs - mins) <= torch.finfo(scores.dtype).eps
    return torch.where(constant, torch.zeros_like(norm), norm)


def context_importance_for_batch(
    teacher: ContextTeacherWorldModel,
    context_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    future_tokens: torch.Tensor,
    *,
    token_chunk_size: int = 32,
    mask_value: float = 0.0,
    clamp_negative: bool = False,
) -> dict[str, torch.Tensor]:
    target = future_target_from_tokens(future_tokens)
    with torch.no_grad():
        base_pred = teacher(context_tokens, current_tokens)
        base_losses = (base_pred - target).pow(2).mean(dim=1)
        masked_losses: list[torch.Tensor] = []
        for start in range(0, context_tokens.shape[1], int(token_chunk_size)):
            end = min(context_tokens.shape[1], start + int(token_chunk_size))
            chunk_losses = []
            for token_idx in range(start, end):
                masked_context = context_tokens.clone()
                masked_context[:, token_idx, :] = float(mask_value)
                pred = teacher(masked_context, current_tokens)
                chunk_losses.append((pred - target).pow(2).mean(dim=1))
            masked_losses.append(torch.stack(chunk_losses, dim=1))
        masked = torch.cat(masked_losses, dim=1)
        importance = masked - base_losses.unsqueeze(1)
        if clamp_negative:
            importance = importance.clamp_min(0.0)
        return {
            "importance_scores": importance,
            "importance_scores_norm": _normalize(importance),
            "base_losses": base_losses,
            "masked_losses": masked,
        }


def _build_dataset(config: dict[str, Any], split: str) -> ContextTokenShardDataset:
    data_cfg = config["data"]
    root = data_cfg[f"{split}_token_shard_dir"]
    max_samples = data_cfg.get(f"max_{split}_samples")
    return ContextTokenShardDataset(
        root,
        shard_glob=data_cfg.get("shard_glob", "context_tokens_shard_*.pt"),
        max_samples=int(max_samples) if max_samples is not None else None,
    )


def _generate_split(config: dict[str, Any], split: str, out_dir: Path, teacher: ContextTeacherWorldModel, checkpoint: dict[str, Any], device: torch.device, overwrite: bool) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.glob("importance_shard_*.pt")) and not overwrite:
        summary_path = out_dir / "eval_importance_summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old in out_dir.glob("importance_shard_*.pt"):
            old.unlink()
    imp_cfg = config["importance"]
    dataset = _build_dataset(config, split)
    loader = DataLoader(dataset, batch_size=int(imp_cfg.get("batch_size", 1)), shuffle=False, num_workers=0, collate_fn=context_token_collate_fn)
    shard_summaries = []
    shard_index = 0
    for batch in loader:
        values = context_importance_for_batch(
            teacher,
            batch["context_tokens"].to(device),
            batch["current_tokens"].to(device),
            batch["future_tokens"].to(device),
            token_chunk_size=int(imp_cfg.get("token_chunk_size", 32)),
            mask_value=float(imp_cfg.get("mask_value", 0.0)),
            clamp_negative=bool(imp_cfg.get("clamp_negative_importance", False)),
        )
        shard = {
            "schema_version": "0.1.0",
            "importance_method": str(imp_cfg.get("method", "context_teacher_token_occlusion")),
            "teacher_checkpoint": str(config["teacher"]["checkpoint"]),
            "teacher_config": dict(checkpoint.get("model_config", {})),
            "source_token_shard": str(batch["metadata"][0].get("token_shard_path", "")),
            "created_at": utc_now_iso(),
            "split": split,
            "sample_ids": batch["sample_ids"],
            "task_texts": batch["task_texts"],
            "importance_scores": values["importance_scores"].detach().cpu(),
            "importance_scores_norm": values["importance_scores_norm"].detach().cpu(),
            "base_losses": values["base_losses"].detach().cpu(),
            "masked_losses": values["masked_losses"].detach().cpu(),
            "metadata": [
                {
                    **dict(meta),
                    "occlude_only": "context_tokens",
                    "keep_current_full": True,
                    "current_tokens_unchanged": True,
                    "trained_current_importance": False,
                }
                for meta in batch["metadata"]
            ],
            "mask_config": {
                "mask_mode": str(imp_cfg.get("mask_mode", "zero")),
                "mask_value": float(imp_cfg.get("mask_value", 0.0)),
                "token_chunk_size": int(imp_cfg.get("token_chunk_size", 32)),
                "occlude_only": "context_tokens",
                "keep_current_full": True,
            },
        }
        path = out_dir / f"importance_shard_{shard_index:06d}.pt"
        save_importance_shard(path, shard)
        summary = summarize_importance_shard(shard)
        summary["path"] = str(path)
        shard_summaries.append(summary)
        print(f"WROTE_CONTEXT_IMPORTANCE_SHARD {path} {summary}", flush=True)
        shard_index += 1
    stats = {
        "split": split,
        "dataset_summary": summarize_context_dataset(dataset, split),
        "num_shards": len(shard_summaries),
        "num_samples": len(dataset),
        "shards": shard_summaries,
        "importance_mean": float(torch.tensor([s["importance_mean"] for s in shard_summaries]).mean().item()) if shard_summaries else None,
        "importance_norm_mean": float(torch.tensor([s["importance_norm_mean"] for s in shard_summaries]).mean().item()) if shard_summaries else None,
        "base_loss_mean": float(torch.tensor([s["base_loss_mean"] for s in shard_summaries]).mean().item()) if shard_summaries else None,
        "masked_loss_mean": float(torch.tensor([s["masked_loss_mean"] for s in shard_summaries]).mean().item()) if shard_summaries else None,
    }
    (out_dir / "eval_importance_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def generate_context_predictive_importance(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    start = time.perf_counter()
    device = resolve_device(str(config["importance"].get("device", "cuda_if_available")))
    teacher, checkpoint = _load_teacher(config["teacher"]["checkpoint"], device)
    output_root = Path(config["output"]["output_root"])
    overwrite = bool(config["output"].get("overwrite", True))
    train = _generate_split(config, "train", Path(config["output"]["train_output_dir"]), teacher, checkpoint, device, overwrite)
    test = _generate_split(config, "test", Path(config["output"]["test_output_dir"]), teacher, checkpoint, device, overwrite)
    summary = {
        "stage": "generate_context_importance_bair_1000_128",
        "output_root": str(output_root),
        "train_num_samples": int(train["num_samples"]),
        "test_num_samples": int(test["num_samples"]),
        "train_num_shards": int(train["num_shards"]),
        "test_num_shards": int(test["num_shards"]),
        "occlude_only": "context_tokens",
        "keep_current_full": True,
        "trained_current_importance": False,
        "train": train,
        "test": test,
        "elapsed_time_sec": round(time.perf_counter() - start, 3),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    Path(config["output"]["summary_path"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "generate_context_importance_bair_1000_128.yaml"))
    return parser.parse_args()


def main() -> None:
    generate_context_predictive_importance(parse_args().config)


if __name__ == "__main__":
    main()
