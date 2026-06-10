"""Generate predictive token importance by teacher token occlusion."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.importance_shards import (
    IMPORTANCE_SHARD_SCHEMA_VERSION,
    save_importance_shard,
    summarize_importance_shard,
    utc_now_iso,
)
from data.token_shards import load_token_shard
from models.teacher_world_model import TeacherWorldModel
from training.teacher_trainer import load_checkpoint, resolve_device, target_from_future_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generate_importance_dummy.yaml")
    parser.add_argument("--token-shard-dir")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
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


def _with_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    updated.setdefault("data", dict(config.get("data", {})))
    updated.setdefault("teacher", dict(config.get("teacher", {})))
    updated.setdefault("importance", dict(config.get("importance", {})))
    updated.setdefault("output", dict(config.get("output", {})))

    updated["data"] = dict(updated["data"])
    updated["teacher"] = dict(updated["teacher"])
    updated["importance"] = dict(updated["importance"])
    updated["output"] = dict(updated["output"])

    if args.token_shard_dir:
        updated["data"]["token_shard_dir"] = args.token_shard_dir
    if args.teacher_checkpoint:
        updated["teacher"]["checkpoint"] = args.teacher_checkpoint
    if args.output_dir:
        updated["output"]["output_dir"] = args.output_dir
        updated["output"]["summary_path"] = str(Path(args.output_dir) / "importance_summary.json")
    if args.device:
        updated["importance"]["device"] = args.device
    if args.max_shards is not None:
        updated["data"]["max_shards"] = int(args.max_shards)
    if args.max_samples is not None:
        updated["data"]["max_samples"] = int(args.max_samples)
    if args.overwrite:
        updated["output"]["overwrite"] = True
    return updated


def _resolve_shard_paths(data_cfg: dict[str, Any]) -> list[Path]:
    shard_dir = Path(data_cfg["token_shard_dir"])
    shard_glob = data_cfg.get("shard_glob", "tokens_shard_*.pt")
    paths = sorted(shard_dir.glob(shard_glob))
    max_shards = data_cfg.get("max_shards")
    if max_shards is not None:
        paths = paths[: int(max_shards)]
    if not paths:
        raise FileNotFoundError(f"No token shards found in {shard_dir} with glob {shard_glob!r}")
    return paths


def _load_teacher(checkpoint_path: str | Path, config_override: dict[str, Any] | None = None) -> tuple[TeacherWorldModel, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = dict(checkpoint.get("model_config") or config_override or {})
    if not model_config:
        raise ValueError("Teacher checkpoint does not contain model_config and no config override was provided")
    model = TeacherWorldModel(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_config


def per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return MSE per sample as shape [B]."""

    if pred.shape != target.shape:
        raise ValueError(f"Prediction shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
    return (pred - target).pow(2).mean(dim=1)


def normalize_importance_scores(scores: torch.Tensor, mode: str = "minmax_per_sample") -> torch.Tensor:
    """Normalize scores without producing NaNs for constant rows."""

    if mode in ("none", None):
        return scores.clone()
    if mode != "minmax_per_sample":
        raise ValueError(f"Unsupported normalization mode: {mode!r}")

    row_min = scores.min(dim=1, keepdim=True).values
    row_max = scores.max(dim=1, keepdim=True).values
    denom = row_max - row_min
    safe = denom > torch.finfo(scores.dtype).eps
    normalized = torch.zeros_like(scores)
    normalized[safe.expand_as(scores)] = (
        (scores - row_min) / denom.clamp_min(torch.finfo(scores.dtype).eps)
    )[safe.expand_as(scores)]
    return normalized


def compute_occlusion_importance(
    model: TeacherWorldModel,
    past_tokens: torch.Tensor,
    future_tokens: torch.Tensor,
    token_chunk_size: int = 32,
    mask_mode: str = "zero",
    mask_value: float = 0.0,
    clamp_negative_importance: bool = False,
    normalize: str = "minmax_per_sample",
) -> dict[str, torch.Tensor]:
    """Compute teacher-token occlusion importance for one token shard batch."""

    if mask_mode != "zero":
        raise ValueError(f"Unsupported mask_mode {mask_mode!r}; Step 5 supports only 'zero'")
    if past_tokens.ndim != 3:
        raise ValueError(f"past_tokens must be [B, N, D], got {tuple(past_tokens.shape)}")
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be at least 1")

    device = next(model.parameters()).device
    past_tokens = past_tokens.to(device=device, dtype=torch.float32)
    future_tokens = future_tokens.to(device=device, dtype=torch.float32)
    target = target_from_future_tokens(future_tokens)

    with torch.no_grad():
        base_pred = model(past_tokens)
        base_losses = per_sample_mse(base_pred, target)
        batch_size, num_tokens, _ = past_tokens.shape
        masked_losses = torch.empty(batch_size, num_tokens, device=device, dtype=torch.float32)

        for start in range(0, num_tokens, token_chunk_size):
            end = min(start + token_chunk_size, num_tokens)
            chunk_size = end - start
            masked = past_tokens.unsqueeze(1).repeat(1, chunk_size, 1, 1)
            for offset, token_index in enumerate(range(start, end)):
                masked[:, offset, token_index, :] = float(mask_value)
            flat_masked = masked.reshape(batch_size * chunk_size, num_tokens, past_tokens.shape[-1])
            repeated_target = target.unsqueeze(1).repeat(1, chunk_size, 1).reshape(
                batch_size * chunk_size,
                target.shape[-1],
            )
            pred_masked = model(flat_masked)
            losses = per_sample_mse(pred_masked, repeated_target).reshape(batch_size, chunk_size)
            masked_losses[:, start:end] = losses

        importance_scores = masked_losses - base_losses.unsqueeze(1)
        if clamp_negative_importance:
            importance_scores = importance_scores.clamp_min(0.0)
        importance_scores_norm = normalize_importance_scores(importance_scores, mode=normalize)

    return {
        "base_losses": base_losses.detach().cpu(),
        "masked_losses": masked_losses.detach().cpu(),
        "importance_scores": importance_scores.detach().cpu(),
        "importance_scores_norm": importance_scores_norm.detach().cpu(),
    }


def _select_max_samples(shard: dict[str, Any], max_samples: int | None) -> dict[str, Any]:
    if max_samples is None:
        return shard
    limit = min(int(max_samples), int(shard["past_tokens"].shape[0]))
    selected = dict(shard)
    selected["past_tokens"] = shard["past_tokens"][:limit]
    selected["future_tokens"] = shard["future_tokens"][:limit]
    selected["sample_ids"] = shard["sample_ids"][:limit]
    selected["task_texts"] = shard["task_texts"][:limit]
    selected["metadata"] = shard["metadata"][:limit]
    return selected


def generate_predictive_importance(config: dict[str, Any]) -> dict[str, Any]:
    """Generate importance shards from token shards and a tiny teacher checkpoint."""

    data_cfg = config["data"]
    teacher_cfg = config["teacher"]
    importance_cfg = config["importance"]
    output_cfg = config["output"]

    output_dir = Path(output_cfg["output_dir"])
    if output_dir.exists() and bool(output_cfg.get("overwrite", False)):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(importance_cfg.get("device", "cuda_if_available")))
    teacher_checkpoint = Path(teacher_cfg["checkpoint"])
    teacher_model, teacher_model_config = _load_teacher(
        teacher_checkpoint,
        config_override=teacher_cfg.get("config"),
    )
    teacher_model.to(device)
    teacher_model.eval()

    shard_paths = _resolve_shard_paths(data_cfg)
    split = str(data_cfg.get("split", "unknown"))
    max_samples = data_cfg.get("max_samples")
    mask_config = {
        "mask_mode": str(importance_cfg.get("mask_mode", "zero")),
        "mask_value": float(importance_cfg.get("mask_value", 0.0)),
        "clamp_negative_importance": bool(importance_cfg.get("clamp_negative_importance", False)),
        "normalize": str(importance_cfg.get("normalize", "minmax_per_sample")),
    }
    token_chunk_size = int(importance_cfg.get("token_chunk_size", 32))
    method = str(importance_cfg.get("method", "teacher_token_occlusion"))

    shard_summaries: list[dict[str, Any]] = []
    output_paths: list[Path] = []
    print(f"teacher checkpoint: {teacher_checkpoint}")
    print(f"source token shard dir: {data_cfg['token_shard_dir']}")
    print(f"output dir: {output_dir}")
    print(f"device: {device}")
    print(f"number of source shards: {len(shard_paths)}")

    for shard_index, shard_path in enumerate(shard_paths):
        token_shard = _select_max_samples(load_token_shard(shard_path, map_location="cpu"), max_samples)
        results = compute_occlusion_importance(
            teacher_model,
            token_shard["past_tokens"],
            token_shard["future_tokens"],
            token_chunk_size=token_chunk_size,
            mask_mode=mask_config["mask_mode"],
            mask_value=mask_config["mask_value"],
            clamp_negative_importance=mask_config["clamp_negative_importance"],
            normalize=mask_config["normalize"],
        )
        importance_shard = {
            "schema_version": IMPORTANCE_SHARD_SCHEMA_VERSION,
            "importance_method": method,
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_config": dict(teacher_model_config),
            "source_token_shard": str(shard_path),
            "created_at": utc_now_iso(),
            "split": split,
            "sample_ids": list(token_shard["sample_ids"]),
            "task_texts": list(token_shard["task_texts"]),
            "importance_scores": results["importance_scores"].cpu(),
            "importance_scores_norm": results["importance_scores_norm"].cpu(),
            "base_losses": results["base_losses"].cpu(),
            "masked_losses": results["masked_losses"].cpu(),
            "metadata": list(token_shard["metadata"]),
            "mask_config": mask_config,
        }
        output_path = output_dir / f"importance_shard_{shard_index:06d}.pt"
        save_importance_shard(output_path, importance_shard)
        summary = summarize_importance_shard(importance_shard)
        summary["path"] = str(output_path)
        shard_summaries.append(summary)
        output_paths.append(output_path)
        print(f"WROTE_IMPORTANCE_SHARD {output_path} {summary}")

    raw_values = torch.cat(
        [torch.as_tensor(summary["importance_mean"]).reshape(1) for summary in shard_summaries]
    )
    summary = {
        "schema_version": IMPORTANCE_SHARD_SCHEMA_VERSION,
        "importance_method": method,
        "teacher_checkpoint": str(teacher_checkpoint),
        "source_token_shard_dir": str(data_cfg["token_shard_dir"]),
        "output_dir": str(output_dir),
        "device": str(device),
        "num_source_shards": len(shard_paths),
        "num_importance_shards": len(output_paths),
        "importance_shard_files": [path.name for path in output_paths],
        "shards": shard_summaries,
        "importance_mean_across_shards": float(raw_values.mean().item()) if raw_values.numel() else 0.0,
    }
    summary_path = Path(output_cfg.get("summary_path", output_dir / "importance_summary.json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"number of importance shards: {len(output_paths)}")
    if shard_summaries:
        first = shard_summaries[0]
        print(f"importance score shape: {first['importance_scores_shape']}")
        print(f"base loss mean: {first['base_loss_mean']}")
        print(f"masked loss mean: {first['masked_loss_mean']}")
        print(
            "importance mean/std/min/max: "
            f"{first['importance_mean']} / {first['importance_std']} / "
            f"{first['importance_min']} / {first['importance_max']}"
        )
    print(f"IMPORTANCE_SUMMARY_WRITTEN = {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    config = _with_cli_overrides(load_yaml(args.config), args)
    summary = generate_predictive_importance(config)
    print("PREDICTIVE_IMPORTANCE_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
