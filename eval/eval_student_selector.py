"""Evaluate a trained Student attention selector on paired importance labels."""

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

from data.student_selector_dataset import StudentSelectorDataset, student_selector_collate_fn
from models.attention_selector import AttentionSelector
from training.student_selector_trainer import evaluate_selector_on_loader, load_student_selector_checkpoint
from training.teacher_trainer import resolve_device
from training.train_student_selector import load_yaml


def evaluate_student_selector(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    train_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    device = resolve_device(str(train_cfg.get("device", "cuda_if_available")))
    checkpoint = load_student_selector_checkpoint(checkpoint_path, map_location="cpu")
    model = AttentionSelector(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if split == "train":
        token_shard_dir = data_cfg.get("train_token_shard_dir", data_cfg.get("token_shard_dir"))
        importance_shard_dir = data_cfg.get("train_importance_shard_dir", data_cfg.get("importance_shard_dir"))
        max_samples = data_cfg.get("max_train_samples", data_cfg.get("max_samples"))
    elif split == "test":
        token_shard_dir = data_cfg.get("test_token_shard_dir", data_cfg.get("token_shard_dir"))
        importance_shard_dir = data_cfg.get("test_importance_shard_dir", data_cfg.get("importance_shard_dir"))
        max_samples = data_cfg.get("max_test_samples", data_cfg.get("max_samples"))
    else:
        raise ValueError(f"Unsupported eval split: {split!r}")
    if token_shard_dir is None:
        raise KeyError(f"data.{split}_token_shard_dir or data.token_shard_dir is required")
    if importance_shard_dir is None:
        raise KeyError(f"data.{split}_importance_shard_dir or data.importance_shard_dir is required")

    dataset = StudentSelectorDataset(
        token_shard_dir=token_shard_dir,
        importance_shard_dir=importance_shard_dir,
        token_shard_glob=data_cfg.get("token_shard_glob", "tokens_shard_*.pt"),
        importance_shard_glob=data_cfg.get("importance_shard_glob", "importance_shard_*.pt"),
        require_key_token_mask=bool(data_cfg.get("require_key_token_mask", True)),
        max_samples=int(max_samples) if max_samples is not None else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=0,
        collate_fn=student_selector_collate_fn,
    )
    metrics = evaluate_selector_on_loader(
        model,
        loader,
        device=device,
        topk=int(train_cfg.get("topk", 4)),
    )
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "split": split,
        "dataset_size": len(dataset),
        "token_shard_dir": str(token_shard_dir),
        "importance_shard_dir": str(importance_shard_dir),
        "device": str(device),
        **metrics,
    }
    run_dir = Path(output_cfg["run_root"]) / output_cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_path = run_dir / "eval_summary.json"
    eval_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["eval_summary_path"] = str(eval_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_student_selector_structured_toy.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_student_selector(load_yaml(args.config), args.checkpoint, split=args.split)
    print("STUDENT_SELECTOR_EVAL_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
