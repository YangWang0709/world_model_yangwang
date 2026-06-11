"""Generate predictive token importance by teacher token occlusion."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
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
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--token-chunk-size", type=int)
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
        if "train_token_shard_dir" in updated["data"] and "test_token_shard_dir" in updated["data"]:
            updated["output"]["output_root"] = args.output_dir
            updated["output"]["train_output_dir"] = str(Path(args.output_dir) / "train")
            updated["output"]["test_output_dir"] = str(Path(args.output_dir) / "test")
            updated["output"]["summary_path"] = str(Path(args.output_dir) / "importance_summary.json")
        else:
            updated["output"]["output_dir"] = args.output_dir
            updated["output"]["summary_path"] = str(Path(args.output_dir) / "importance_summary.json")
    if args.device:
        updated["importance"]["device"] = args.device
    if args.max_shards is not None:
        updated["data"]["max_shards"] = int(args.max_shards)
    if args.max_samples is not None:
        updated["data"]["max_samples"] = int(args.max_samples)
    if args.batch_size is not None:
        updated["importance"]["batch_size"] = int(args.batch_size)
    if args.token_chunk_size is not None:
        updated["importance"]["token_chunk_size"] = int(args.token_chunk_size)
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


def resolve_teacher_checkpoint(path: str | Path) -> Path:
    """Resolve a configured checkpoint path, falling back to the latest run checkpoint."""

    checkpoint_path = Path(path)
    if checkpoint_path.exists():
        return checkpoint_path
    checkpoint_dir = checkpoint_path.parent
    candidates = sorted(checkpoint_dir.glob("teacher_world_model_step_*.pt"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")


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


def _is_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


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
    return _select_sample_range(shard, start=0, end=min(int(max_samples), int(shard["past_tokens"].shape[0])))


def _select_sample_range(shard: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    """Return a shallow token shard slice with tensors and per-sample lists aligned."""

    if start < 0 or end < start:
        raise ValueError(f"Invalid sample range: start={start}, end={end}")
    selected = dict(shard)
    selected["past_tokens"] = shard["past_tokens"][start:end]
    selected["future_tokens"] = shard["future_tokens"][start:end]
    selected["sample_ids"] = shard["sample_ids"][start:end]
    selected["task_texts"] = shard["task_texts"][start:end]
    selected["metadata"] = shard["metadata"][start:end]
    return selected


def _generate_predictive_importance_single(
    config: dict[str, Any],
    data_cfg_override: dict[str, Any] | None = None,
    output_cfg_override: dict[str, Any] | None = None,
    teacher_bundle: tuple[TeacherWorldModel, dict[str, Any], Path, torch.device] | None = None,
) -> dict[str, Any]:
    """Generate importance shards for one token-shard directory."""

    start_time = time.perf_counter()

    data_cfg = data_cfg_override or config["data"]
    teacher_cfg = config["teacher"]
    importance_cfg = config["importance"]
    output_cfg = output_cfg_override or config["output"]

    output_dir = Path(output_cfg["output_dir"])
    if output_dir.exists() and bool(output_cfg.get("overwrite", False)):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if teacher_bundle is None:
        device = resolve_device(str(importance_cfg.get("device", "cuda_if_available")))
        teacher_checkpoint = resolve_teacher_checkpoint(teacher_cfg["checkpoint"])
        teacher_model, teacher_model_config = _load_teacher(
            teacher_checkpoint,
            config_override=teacher_cfg.get("config"),
        )
        teacher_model.to(device)
        teacher_model.eval()
    else:
        teacher_model, teacher_model_config, teacher_checkpoint, device = teacher_bundle

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
    batch_size = int(importance_cfg.get("batch_size", 0) or 0)
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    method = str(importance_cfg.get("method", "teacher_token_occlusion"))

    shard_summaries: list[dict[str, Any]] = []
    output_paths: list[Path] = []
    total_samples = 0
    observed_num_tokens: int | None = None
    observed_token_dim: int | None = None
    source_encoder = "unknown"
    source_dataset = "unknown"
    source_split = split
    print(f"teacher checkpoint: {teacher_checkpoint}")
    print(f"source token shard dir: {data_cfg['token_shard_dir']}")
    print(f"output dir: {output_dir}")
    print(f"device: {device}")
    print(f"number of source shards: {len(shard_paths)}")

    try:
        for source_shard_index, shard_path in enumerate(shard_paths):
            raw_shard = load_token_shard(shard_path, map_location="cpu")
            if source_shard_index == 0:
                source_encoder = str(raw_shard.get("encoder_name", "unknown"))
                source_split = str(raw_shard.get("split", split))
                metadata = raw_shard.get("metadata") or []
                first_metadata = metadata[0] if metadata else {}
                if isinstance(first_metadata, dict):
                    source_dataset = str(first_metadata.get("source", first_metadata.get("dataset", "unknown")))
            shard_batch = int(raw_shard["past_tokens"].shape[0])
            remaining = None if max_samples is None else int(max_samples) - total_samples
            if remaining is not None and remaining <= 0:
                break
            take = shard_batch if remaining is None else min(shard_batch, remaining)
            selected_shard = _select_max_samples(raw_shard, take)
            effective_batch_size = batch_size or int(selected_shard["past_tokens"].shape[0])
            if effective_batch_size < 1:
                raise ValueError("effective batch size must be at least 1")

            for batch_start in range(0, int(selected_shard["past_tokens"].shape[0]), effective_batch_size):
                batch_end = min(batch_start + effective_batch_size, int(selected_shard["past_tokens"].shape[0]))
                token_shard = _select_sample_range(selected_shard, batch_start, batch_end)
                if int(token_shard["past_tokens"].shape[0]) == 0:
                    continue
                batch_num_tokens = int(token_shard["past_tokens"].shape[1])
                batch_token_dim = int(token_shard["past_tokens"].shape[2])
                observed_num_tokens = observed_num_tokens or batch_num_tokens
                observed_token_dim = observed_token_dim or batch_token_dim

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
                    "source_batch_range": {
                        "source_shard_index": source_shard_index,
                        "start": batch_start,
                        "end": batch_end,
                    },
                }
                output_path = output_dir / f"importance_shard_{len(output_paths):06d}.pt"
                save_importance_shard(output_path, importance_shard)
                summary = summarize_importance_shard(importance_shard)
                summary["path"] = str(output_path)
                summary["source_shard_index"] = source_shard_index
                summary["source_batch_range"] = [batch_start, batch_end]
                shard_summaries.append(summary)
                output_paths.append(output_path)
                total_samples += int(results["importance_scores"].shape[0])
                print(f"WROTE_IMPORTANCE_SHARD {output_path} {summary}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except RuntimeError as exc:
        if _is_oom_error(exc):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM while generating predictive importance; reduce token_chunk_size "
                "or move the next run to a larger 4090 / 48GB server."
            ) from exc
        raise

    if not output_paths:
        raise RuntimeError("No importance shards were generated")
    raw_values = torch.cat(
        [torch.as_tensor(summary["importance_mean"]).reshape(1) for summary in shard_summaries]
    )
    elapsed_time_sec = round(time.perf_counter() - start_time, 3)
    summary = {
        "schema_version": IMPORTANCE_SHARD_SCHEMA_VERSION,
        "importance_method": method,
        "teacher_checkpoint": str(teacher_checkpoint),
        "source_token_shard_dir": str(data_cfg["token_shard_dir"]),
        "output_dir": str(output_dir),
        "split": split,
        "source_encoder": source_encoder,
        "source_split": source_split,
        "dataset": source_dataset,
        "device": str(device),
        "num_source_shards": len(shard_paths),
        "num_importance_shards": len(output_paths),
        "num_samples": int(total_samples),
        "num_tokens": int(observed_num_tokens or 0),
        "token_dim": int(observed_token_dim or int(teacher_model_config.get("token_dim", 0) or 0)),
        "batch_size": batch_size,
        "token_chunk_size": token_chunk_size,
        "max_samples": int(max_samples) if max_samples is not None else None,
        "mask_config": mask_config,
        "importance_shard_files": [path.name for path in output_paths],
        "shards": shard_summaries,
        "importance_mean_across_shards": float(raw_values.mean().item()) if raw_values.numel() else 0.0,
        "elapsed_time_sec": elapsed_time_sec,
        "oom": False,
        "resource_limits": dict(config.get("resource_limits", {})),
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


def _has_train_test_split_dirs(config: dict[str, Any]) -> bool:
    data_cfg = config.get("data", {})
    return "train_token_shard_dir" in data_cfg and "test_token_shard_dir" in data_cfg


def _combine_split_generation_summaries(
    config: dict[str, Any],
    train_summary: dict[str, Any],
    test_summary: dict[str, Any],
    output_root: Path,
    summary_path: Path,
    elapsed_time_sec: float,
) -> dict[str, Any]:
    importance_cfg = config["importance"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    split_summaries = {"train": train_summary, "test": test_summary}
    total_samples = int(train_summary["num_samples"]) + int(test_summary["num_samples"])
    total_shards = int(train_summary["num_importance_shards"]) + int(test_summary["num_importance_shards"])
    shard_means: list[float] = []
    for split_summary in split_summaries.values():
        shard_means.extend(float(item["importance_mean"]) for item in split_summary.get("shards", []))

    summary = {
        "schema_version": IMPORTANCE_SHARD_SCHEMA_VERSION,
        "importance_method": str(importance_cfg.get("method", "teacher_token_occlusion")),
        "teacher_checkpoint": str(train_summary["teacher_checkpoint"]),
        "output_root": str(output_root),
        "train_output_dir": str(output_cfg["train_output_dir"]),
        "test_output_dir": str(output_cfg["test_output_dir"]),
        "train_token_shard_dir": str(data_cfg["train_token_shard_dir"]),
        "test_token_shard_dir": str(data_cfg["test_token_shard_dir"]),
        "device": str(train_summary["device"]),
        "num_importance_shards": total_shards,
        "num_samples": total_samples,
        "train_num_samples": int(train_summary["num_samples"]),
        "test_num_samples": int(test_summary["num_samples"]),
        "num_tokens": int(train_summary["num_tokens"]),
        "token_dim": int(train_summary["token_dim"]),
        "source_encoder": str(train_summary.get("source_encoder", "unknown")),
        "dataset": str(train_summary.get("dataset", "unknown")),
        "batch_size": int(importance_cfg.get("batch_size", 0) or 0),
        "token_chunk_size": int(importance_cfg.get("token_chunk_size", 32)),
        "max_train_samples": int(data_cfg["max_train_samples"]) if data_cfg.get("max_train_samples") is not None else None,
        "max_test_samples": int(data_cfg["max_test_samples"]) if data_cfg.get("max_test_samples") is not None else None,
        "mask_config": dict(train_summary["mask_config"]),
        "split_summaries": split_summaries,
        "importance_mean_across_shards": (
            float(torch.tensor(shard_means, dtype=torch.float32).mean().item()) if shard_means else 0.0
        ),
        "elapsed_time_sec": elapsed_time_sec,
        "oom": False,
        "resource_limits": dict(config.get("resource_limits", {})),
        "summary_path": str(summary_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def generate_predictive_importance(config: dict[str, Any]) -> dict[str, Any]:
    """Generate importance shards from token shards and a teacher checkpoint."""

    if not _has_train_test_split_dirs(config):
        return _generate_predictive_importance_single(config)

    start_time = time.perf_counter()
    data_cfg = dict(config["data"])
    output_cfg = config["output"]
    importance_cfg = config["importance"]
    teacher_cfg = config["teacher"]
    output_root = Path(output_cfg["output_root"])
    if output_root.exists() and bool(output_cfg.get("overwrite", False)):
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(importance_cfg.get("device", "cuda_if_available")))
    teacher_checkpoint = resolve_teacher_checkpoint(teacher_cfg["checkpoint"])
    teacher_model, teacher_model_config = _load_teacher(
        teacher_checkpoint,
        config_override=teacher_cfg.get("config"),
    )
    teacher_model.to(device)
    teacher_model.eval()
    teacher_bundle = (teacher_model, teacher_model_config, teacher_checkpoint, device)

    train_data_cfg = dict(data_cfg)
    train_data_cfg["token_shard_dir"] = data_cfg["train_token_shard_dir"]
    train_data_cfg["split"] = str(data_cfg.get("split_train", "train"))
    if data_cfg.get("max_train_samples") is not None:
        train_data_cfg["max_samples"] = int(data_cfg["max_train_samples"])

    test_data_cfg = dict(data_cfg)
    test_data_cfg["token_shard_dir"] = data_cfg["test_token_shard_dir"]
    test_data_cfg["split"] = str(data_cfg.get("split_test", "test"))
    if data_cfg.get("max_test_samples") is not None:
        test_data_cfg["max_samples"] = int(data_cfg["max_test_samples"])

    train_output_cfg = dict(output_cfg)
    train_output_cfg["output_dir"] = output_cfg["train_output_dir"]
    train_output_cfg["summary_path"] = str(Path(output_cfg["train_output_dir"]) / "importance_summary.json")
    train_output_cfg["overwrite"] = False

    test_output_cfg = dict(output_cfg)
    test_output_cfg["output_dir"] = output_cfg["test_output_dir"]
    test_output_cfg["summary_path"] = str(Path(output_cfg["test_output_dir"]) / "importance_summary.json")
    test_output_cfg["overwrite"] = False

    train_summary = _generate_predictive_importance_single(
        config,
        data_cfg_override=train_data_cfg,
        output_cfg_override=train_output_cfg,
        teacher_bundle=teacher_bundle,
    )
    test_summary = _generate_predictive_importance_single(
        config,
        data_cfg_override=test_data_cfg,
        output_cfg_override=test_output_cfg,
        teacher_bundle=teacher_bundle,
    )
    elapsed_time_sec = round(time.perf_counter() - start_time, 3)
    summary_path = Path(output_cfg.get("summary_path", output_root / "importance_summary.json"))
    summary = _combine_split_generation_summaries(
        config,
        train_summary=train_summary,
        test_summary=test_summary,
        output_root=output_root,
        summary_path=summary_path,
        elapsed_time_sec=elapsed_time_sec,
    )
    print(f"number of train importance shards: {train_summary['num_importance_shards']}")
    print(f"number of test importance shards: {test_summary['num_importance_shards']}")
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
