"""Evaluate a Step 7 Student world model checkpoint."""

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

from data.student_selector_dataset import student_selector_collate_fn
from eval.eval_efficiency import token_retention_ratio
from training.student_world_model_trainer import (
    build_student_world_model_dataset,
    build_student_world_model_bundle,
    evaluate_student_world_model_on_loader,
)
from training.teacher_trainer import resolve_device
from training.train_student_world_model import load_yaml


def future_mse_value(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.shape != target.shape:
        raise ValueError(f"Prediction shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
    return float(torch.nn.functional.mse_loss(pred, target).item())


def evaluate_student_world_model(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    train_cfg = config["training"]
    selection_cfg = config.get("selection", {})
    output_cfg = config["output"]
    device = resolve_device(str(train_cfg.get("device", "cuda_if_available")))
    selector, compressor, student_world_model, checkpoint = build_student_world_model_bundle(checkpoint_path, device)

    dataset = build_student_world_model_dataset(config, split=split)
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=0,
        collate_fn=student_selector_collate_fn,
    )
    topk = int(selection_cfg.get("topk", train_cfg.get("topk", checkpoint.get("metrics_summary", {}).get("topk", 4))))
    use_sigmoid_scores = bool(
        selection_cfg.get("use_sigmoid_scores", train_cfg.get("use_sigmoid_scores", True))
    )
    metrics = evaluate_student_world_model_on_loader(
        selector,
        compressor,
        student_world_model,
        loader,
        device=device,
        topk=topk,
        use_sigmoid_scores=use_sigmoid_scores,
    )
    summary = {
        "split": split,
        "checkpoint_path": str(checkpoint_path),
        "selector_checkpoint_path": checkpoint["selector_checkpoint_path"],
        "selector_frozen": bool(checkpoint.get("selector_frozen", True)),
        "dataset_size": len(dataset),
        "num_samples": len(dataset),
        "device": str(device),
        "topk": topk,
        "num_tokens": int(dataset[0]["past_tokens"].shape[0]),
        "token_dim": int(dataset[0]["past_tokens"].shape[1]),
        "token_retention_ratio": token_retention_ratio(
            selected_tokens=topk,
            total_tokens=int(dataset[0]["past_tokens"].shape[0]),
        ),
        "compressed_tokens": int(checkpoint["compressor_config"]["num_latents"]),
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
    parser.add_argument("--config", default="configs/train_student_world_model_structured_toy.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_student_world_model(load_yaml(args.config), args.checkpoint, split=args.split)
    print("STUDENT_WORLD_MODEL_EVAL_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
