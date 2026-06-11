"""Evaluate a tiny full-token teacher checkpoint on token shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.token_shard_dataset import TokenShardDataset, token_shard_collate_fn
from models.teacher_world_model import TeacherWorldModel
from training.losses import future_latent_mse
from training.teacher_trainer import load_checkpoint, resolve_device, summarize_token_dataset, target_from_future_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_teacher_dummy.yaml")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def evaluate_teacher(config: dict[str, Any], checkpoint_path: str | Path) -> dict[str, Any]:
    train_cfg = config["training"]
    device = resolve_device(str(train_cfg.get("device", "cuda_if_available")))
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    model = TeacherWorldModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_cfg = config["data"]
    output_cfg = config["output"]
    token_shard_dir = (
        data_cfg.get("eval_token_shard_dir")
        or data_cfg.get("test_token_shard_dir")
        or data_cfg.get("token_shard_dir")
    )
    if token_shard_dir is None:
        raise KeyError("data.token_shard_dir, data.test_token_shard_dir, or data.eval_token_shard_dir is required")
    max_samples_value = (
        data_cfg.get("max_eval_samples")
        or data_cfg.get("max_test_samples")
        or data_cfg.get("max_samples")
    )
    dataset = TokenShardDataset(
        token_shard_dir,
        shard_glob=data_cfg.get("shard_glob", "tokens_shard_*.pt"),
        map_location="cpu",
        max_samples=int(max_samples_value) if max_samples_value is not None else None,
    )
    split = str(data_cfg.get("split_eval", data_cfg.get("split_test", data_cfg.get("split", "")))) or None
    dataset_summary = summarize_token_dataset(dataset, split=split)
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=0,
        collate_fn=token_shard_collate_fn,
    )

    losses: list[float] = []
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            past_tokens = batch["past_tokens"].to(device)
            future_tokens = batch["future_tokens"].to(device)
            pred = model(past_tokens)
            target = target_from_future_tokens(future_tokens)
            if pred.shape != target.shape:
                raise ValueError(f"Prediction shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
            losses.append(float(future_latent_mse(pred, target).item()))
            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())

    eval_mse = float(sum(losses) / max(1, len(losses)))
    pred_tensor = torch.cat(preds, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    run_dir = Path(output_cfg["run_root"]) / output_cfg["run_name"]
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "dataset": dataset_summary.get("dataset", "unknown"),
        "split": split or dataset_summary["source_split"],
        "dataset_size": len(dataset),
        "num_samples": len(dataset),
        "num_tokens": dataset_summary["num_tokens"],
        "token_dim": dataset_summary["token_dim"],
        "source_encoder": dataset_summary["source_encoder"],
        "source_split": dataset_summary["source_split"],
        "token_shard_dir": dataset_summary["token_shard_dir"],
        "num_batches": len(losses),
        "eval_mse": eval_mse,
        "pred_mean": float(pred_tensor.mean().item()),
        "pred_std": float(pred_tensor.std(unbiased=False).item()),
        "pred_min": float(pred_tensor.min().item()),
        "pred_max": float(pred_tensor.max().item()),
        "target_mean": float(target_tensor.mean().item()),
        "target_std": float(target_tensor.std(unbiased=False).item()),
        "target_min": float(target_tensor.min().item()),
        "target_max": float(target_tensor.max().item()),
        "device": str(device),
        "run_dir": str(run_dir),
    }
    output_path = run_dir / "eval_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"EVAL_SUMMARY_WRITTEN = {output_path}")
    return summary


def main() -> None:
    args = parse_args()
    evaluate_teacher(load_yaml(args.config), args.checkpoint)


if __name__ == "__main__":
    main()
