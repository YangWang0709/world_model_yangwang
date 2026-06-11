"""Compare full-token Teacher and compressed Student future prediction."""

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
from models.teacher_world_model import TeacherWorldModel
from training.student_world_model_trainer import (
    build_student_world_model_dataset,
    build_student_world_model_bundle,
    selection_quality_metrics,
    student_world_model_forward,
)
from training.teacher_trainer import load_checkpoint, resolve_device, target_from_future_tokens
from training.train_student_world_model import load_yaml


def compute_teacher_student_gap(
    teacher_future_mse: float,
    student_future_mse: float,
    eps: float = 1e-12,
) -> dict[str, float]:
    if teacher_future_mse < 0.0 or student_future_mse < 0.0:
        raise ValueError("MSE values must be non-negative")
    denominator = max(float(teacher_future_mse), float(eps))
    return {
        "student_teacher_gap": float(student_future_mse) - float(teacher_future_mse),
        "student_teacher_ratio": float(student_future_mse) / denominator,
    }


def _load_teacher(checkpoint_path: str | Path, device: torch.device) -> TeacherWorldModel:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model = TeacherWorldModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_teacher_student_gap(
    config: dict[str, Any],
    student_checkpoint_path: str | Path,
    teacher_checkpoint_path: str | Path | None = None,
    split: str = "test",
) -> dict[str, Any]:
    train_cfg = config["training"]
    selection_cfg = config.get("selection", {})
    data_cfg = config["data"]
    output_cfg = config["output"]
    teacher_cfg = config.get("teacher_reference", {})
    resolved_teacher_checkpoint = str(teacher_checkpoint_path or teacher_cfg["checkpoint"])

    device = resolve_device(str(train_cfg.get("device", "cuda_if_available")))
    teacher = _load_teacher(resolved_teacher_checkpoint, device)
    selector, compressor, student_world_model, student_checkpoint = build_student_world_model_bundle(
        student_checkpoint_path,
        device,
    )

    dataset = build_student_world_model_dataset(config, split=split)
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=0,
        collate_fn=student_selector_collate_fn,
    )
    topk = int(selection_cfg.get("topk", train_cfg.get("topk", student_checkpoint.get("metrics_summary", {}).get("topk", 4))))
    use_sigmoid_scores = bool(selection_cfg.get("use_sigmoid_scores", train_cfg.get("use_sigmoid_scores", True)))

    teacher_sse = 0.0
    student_sse = 0.0
    element_count = 0
    all_scores = []
    all_masks = []
    all_importance_targets = []
    with torch.no_grad():
        for batch in loader:
            past_tokens = batch["past_tokens"].to(device)
            future_tokens = batch["future_tokens"].to(device)
            target = target_from_future_tokens(future_tokens)
            teacher_pred = teacher(past_tokens)
            student_pred, scores, _, _ = student_world_model_forward(
                selector,
                compressor,
                student_world_model,
                past_tokens,
                topk=topk,
                use_sigmoid_scores=use_sigmoid_scores,
            )
            if teacher_pred.shape != target.shape:
                raise ValueError(
                    f"Teacher prediction shape {tuple(teacher_pred.shape)} != target shape {tuple(target.shape)}"
                )
            if student_pred.shape != target.shape:
                raise ValueError(
                    f"Student prediction shape {tuple(student_pred.shape)} != target shape {tuple(target.shape)}"
                )
            teacher_sse += float((teacher_pred - target).pow(2).sum().item())
            student_sse += float((student_pred - target).pow(2).sum().item())
            element_count += int(target.numel())
            all_scores.append(scores.detach().cpu())
            key_token_mask = batch.get("key_token_mask")
            if key_token_mask is not None:
                all_masks.append(key_token_mask.float().cpu())
            all_importance_targets.append(batch["importance_scores_norm"].float().cpu())

    teacher_mse = teacher_sse / float(max(element_count, 1))
    student_mse = student_sse / float(max(element_count, 1))
    score_tensor = torch.cat(all_scores, dim=0)
    mask_tensor = torch.cat(all_masks, dim=0) if all_masks else None
    importance_tensor = torch.cat(all_importance_targets, dim=0) if all_importance_targets else None
    selection_metrics = selection_quality_metrics(score_tensor, mask_tensor, importance_tensor, k=topk)
    gap_metrics = compute_teacher_student_gap(teacher_mse, student_mse)
    retention = token_retention_ratio(selected_tokens=topk, total_tokens=int(score_tensor.shape[1]))
    summary = {
        "dataset": str(data_cfg.get("dataset", dataset[0].get("metadata", {}).get("dataset", "unknown"))),
        "split": split,
        "student_checkpoint_path": str(student_checkpoint_path),
        "teacher_checkpoint_path": resolved_teacher_checkpoint,
        "dataset_size": len(dataset),
        "num_samples": len(dataset),
        "device": str(device),
        "teacher_future_mse": teacher_mse,
        "student_future_mse": student_mse,
        "teacher_mse": teacher_mse,
        "student_mse": student_mse,
        "token_retention_ratio": retention,
        "topk": topk,
        "total_tokens": int(score_tensor.shape[1]),
        "num_tokens": int(score_tensor.shape[1]),
        "compressed_tokens": int(student_checkpoint["compressor_config"]["num_latents"]),
        **gap_metrics,
        **selection_metrics,
    }

    run_dir = Path(output_cfg["run_root"]) / output_cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    gap_path = run_dir / "teacher_student_gap_summary.json"
    gap_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["gap_summary_path"] = str(gap_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_student_world_model_structured_toy.yaml")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_teacher_student_gap(
        load_yaml(args.config),
        student_checkpoint_path=args.student_checkpoint,
        teacher_checkpoint_path=args.teacher_checkpoint,
        split=args.split,
    )
    print("TEACHER_STUDENT_GAP_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
