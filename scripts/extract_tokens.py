"""Offline frozen video token extraction pipeline."""

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

from data.bair_dataset import BAIRRobotPushingDataset
from data.real_video_dataset import RealVideoClipDataset
from data.token_shards import save_token_shard, summarize_token_shard, utc_now_iso
from data.video_clip_dataset import VideoClipDataset
from encoders.frozen_video_encoder import build_frozen_video_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/token_extraction_dummy.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_device(device_name: str | None) -> torch.device:
    if device_name is None or device_name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _metadata_for_item(item: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(item.get("metadata", {}))
    metadata.setdefault("sample_id", str(item.get("sample_id", "")))
    if "actions" in item:
        actions = item.get("actions")
        metadata["has_action"] = isinstance(actions, torch.Tensor)
        metadata["action_shape"] = list(actions.shape) if isinstance(actions, torch.Tensor) else None
        if str(metadata.get("source", "")).startswith("bair") or str(item.get("sample_id", "")).startswith("bair_"):
            metadata.setdefault("source", "bair_robot_pushing_small")
    if "endeffector_pos" in item:
        endeffector_pos = item.get("endeffector_pos")
        metadata["has_endeffector_pos"] = isinstance(endeffector_pos, torch.Tensor)
        metadata["endeffector_pos_shape"] = (
            list(endeffector_pos.shape) if isinstance(endeffector_pos, torch.Tensor) else None
        )
    return metadata


def collate_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "past_video": torch.stack([item["past_video"] for item in batch], dim=0),
        "future_video": torch.stack([item["future_video"] for item in batch], dim=0),
        "task_text": [item["task_text"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "metadata": [_metadata_for_item(item) for item in batch],
    }


def _resolve_effective_max_samples(
    dataset_cfg: dict[str, Any],
    extraction_cfg: dict[str, Any],
    cli_max_samples: int | None,
    split: str | None = None,
) -> int | None:
    if cli_max_samples is not None:
        return cli_max_samples
    if split:
        split_key = f"max_{split}_samples"
        if dataset_cfg.get(split_key) is not None:
            return int(dataset_cfg[split_key])
    if extraction_cfg.get("max_samples") is not None:
        return int(extraction_cfg["max_samples"])
    if dataset_cfg.get("max_samples") is not None:
        return int(dataset_cfg["max_samples"])
    return None


def _build_dataset(
    dataset_cfg: dict[str, Any],
    extraction_cfg: dict[str, Any],
    max_samples: int | None,
    split: str | None = None,
) -> Any:
    dataset_name = str(dataset_cfg.get("name", "toy_videos"))
    if dataset_name == "toy_videos":
        dataset = VideoClipDataset(
            root=resolve_path(dataset_cfg["root"]),
            metadata_file=dataset_cfg.get("metadata_file", "metadata.jsonl"),
            past_len=int(dataset_cfg.get("past_len", 4)),
            future_len=int(dataset_cfg.get("future_len", 2)),
        )
        if max_samples is not None:
            dataset.records = dataset.records[:max_samples]
        return dataset
    if dataset_name == "real_video_minimal":
        return RealVideoClipDataset(
            root=resolve_path(dataset_cfg["root"]),
            metadata_file=dataset_cfg.get("metadata_file", "metadata.jsonl"),
            past_len=int(dataset_cfg.get("past_len", 4)),
            future_len=int(dataset_cfg.get("future_len", 4)),
            image_size=int(dataset_cfg.get("image_size", 224)),
            split=dataset_cfg.get("split"),
            max_samples=max_samples,
        )
    if dataset_name == "bair_robot_pushing_small_subset":
        split_name = split or str(dataset_cfg.get("split", "train"))
        metadata_value = dataset_cfg.get(f"{split_name}_metadata_file") or dataset_cfg.get("metadata_file")
        return BAIRRobotPushingDataset(
            root=resolve_path(dataset_cfg["root"]),
            split=split_name,
            metadata_file=metadata_value,
            past_len=int(dataset_cfg.get("past_len", 4)),
            future_len=int(dataset_cfg.get("future_len", 4)),
            image_size=int(dataset_cfg.get("image_size", 224)),
            max_samples=max_samples,
        )
    raise ValueError(f"Unsupported dataset.name {dataset_name!r}")


def _build_encoder(
    encoder_cfg: dict[str, Any],
    fallback_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[Any, str, bool, str | None, dict[str, Any]]:
    requested_encoder = str(encoder_cfg.get("name", "dummy_video_encoder"))
    encoder_build_cfg = dict(encoder_cfg)
    encoder_build_cfg.setdefault("device", str(device))
    frozen_encoder = build_frozen_video_encoder(encoder_build_cfg)
    used_fallback = False
    fallback_reason = None
    encoder_availability = frozen_encoder.availability
    if frozen_encoder.is_available():
        return frozen_encoder, frozen_encoder.encoder_name, used_fallback, fallback_reason, encoder_availability
    if bool(fallback_cfg.get("allow_dummy_fallback", False)):
        fallback_reason = str(encoder_availability.get("reason", "requested frozen encoder unavailable"))
        print(
            "FROZEN_ENCODER_FALLBACK "
            f"requested={requested_encoder} actual=dummy_video_encoder reason={fallback_reason}"
        )
        encoder = build_frozen_video_encoder(
            {
                "name": "dummy_video_encoder",
                "num_tokens": int(fallback_cfg.get("dummy_num_tokens", encoder_cfg.get("num_tokens", 196))),
                "token_dim": int(fallback_cfg.get("dummy_token_dim", encoder_cfg.get("token_dim", 768))),
                "device": str(device),
            }
        )
        return encoder, "dummy_video_encoder", True, fallback_reason, encoder_availability
    raise RuntimeError(
        f"Requested encoder {requested_encoder!r} is unavailable and fallback is disabled: "
        f"{encoder_availability.get('reason', 'unknown reason')}"
    )


def _encoder_runtime_summary(encoder: Any) -> dict[str, Any]:
    impl = getattr(encoder, "impl", encoder)
    summary = dict(getattr(impl, "last_encode_summary", {}) or {})
    if summary:
        input_shape = summary.get("input_shape") or []
        effective_shape = summary.get("effective_input_shape") or []
        if len(input_shape) >= 2 and len(effective_shape) >= 2 and input_shape[1] != effective_shape[1]:
            summary["temporal_pad_to"] = int(effective_shape[1])
    return summary


def _extract_dataset_to_shards(
    *,
    config_path: Path,
    dataset: Any,
    dataset_cfg: dict[str, Any],
    encoder_cfg: dict[str, Any],
    extraction_cfg: dict[str, Any],
    fallback_cfg: dict[str, Any],
    encoder: Any,
    actual_encoder: str,
    used_fallback: bool,
    fallback_reason: str | None,
    encoder_availability: dict[str, Any],
    out_dir: Path,
    device: torch.device,
    effective_max_samples: int | None,
    split_name: str,
    start_time: float,
    overwrite: bool = False,
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.glob("tokens_shard_*.pt")) and not overwrite:
        raise FileExistsError(f"Output dir already contains token shards. Use --overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old_shard in out_dir.glob("tokens_shard_*.pt"):
            old_shard.unlink()
        old_summary = out_dir / "extraction_summary.json"
        if old_summary.exists():
            old_summary.unlink()

    batch_size = int(extraction_cfg.get("batch_size", 4))
    shard_size = int(extraction_cfg.get("shard_size", 8))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(extraction_cfg.get("num_workers", 0)),
        collate_fn=collate_samples,
    )
    requested_encoder = str(encoder_cfg.get("name", "dummy_video_encoder"))

    shard_buffer: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    shard_index = 0
    observed_runtime_summary: dict[str, Any] = {}

    def flush_shard() -> None:
        nonlocal shard_buffer, shard_index
        if not shard_buffer:
            return

        past_tokens = torch.cat([item["past_tokens"] for item in shard_buffer], dim=0).cpu()
        future_tokens = torch.cat([item["future_tokens"] for item in shard_buffer], dim=0).cpu()
        sample_ids = [sample_id for item in shard_buffer for sample_id in item["sample_ids"]]
        task_texts = [task_text for item in shard_buffer for task_text in item["task_texts"]]
        metadata = [metadata for item in shard_buffer for metadata in item["metadata"]]
        shard = {
            "schema_version": "0.1.0",
            "encoder_name": actual_encoder,
            "encoder_config": {
                "requested_encoder": requested_encoder,
                "actual_encoder": actual_encoder,
                "used_fallback": used_fallback,
                "fallback_reason": fallback_reason,
                "model_name_or_path": encoder_cfg.get("model_name_or_path"),
                "cache_dir": encoder_cfg.get("cache_dir"),
                "output_mode": encoder_cfg.get("output_mode"),
                "image_size": encoder_cfg.get("image_size"),
                "num_frames": encoder_cfg.get("num_frames"),
                "token_dim": int(past_tokens.shape[-1]),
                "num_tokens": int(past_tokens.shape[1]),
                "dummy_num_tokens": int(fallback_cfg.get("dummy_num_tokens", encoder_cfg.get("num_tokens", 196))),
                "dummy_token_dim": int(fallback_cfg.get("dummy_token_dim", encoder_cfg.get("token_dim", 768))),
                "patch_grid_h": int(encoder_cfg.get("patch_grid_h", 14)),
                "patch_grid_w": int(encoder_cfg.get("patch_grid_w", 14)),
                "encoder_availability": encoder_availability,
                "runtime_summary": dict(observed_runtime_summary),
            },
            "created_at": utc_now_iso(),
            "split": split_name,
            "sample_ids": sample_ids,
            "task_texts": task_texts,
            "past_tokens": past_tokens,
            "future_tokens": future_tokens,
            "metadata": metadata,
        }
        temporal_pad_to = observed_runtime_summary.get("temporal_pad_to")
        if temporal_pad_to is not None:
            shard["encoder_config"]["temporal_pad_to"] = int(temporal_pad_to)
        shard_path = out_dir / f"tokens_shard_{shard_index:06d}.pt"
        save_token_shard(shard_path, shard)
        summary = summarize_token_shard(shard)
        summary["path"] = str(shard_path)
        shard_summaries.append(summary)
        print(f"WROTE_SHARD {shard_path} {summary}")
        shard_index += 1
        shard_buffer = []

    with torch.no_grad():
        for batch in loader:
            past_video = batch["past_video"]
            future_video = batch["future_video"]
            past_tokens = encoder.encode(past_video)
            observed_runtime_summary = observed_runtime_summary or _encoder_runtime_summary(encoder)
            future_tokens = encoder.encode(future_video)
            observed_runtime_summary = observed_runtime_summary or _encoder_runtime_summary(encoder)
            shard_buffer.append(
                {
                    "past_tokens": past_tokens.cpu(),
                    "future_tokens": future_tokens.cpu(),
                    "sample_ids": list(batch["sample_id"]),
                    "task_texts": list(batch["task_text"]),
                    "metadata": list(batch["metadata"]),
                }
            )

            buffered_samples = sum(item["past_tokens"].shape[0] for item in shard_buffer)
            if buffered_samples >= shard_size:
                flush_shard()

    flush_shard()

    summary = {
        "config": str(config_path),
        "dataset_size": len(dataset),
        "batch_size": batch_size,
        "shard_size": shard_size,
        "max_samples": effective_max_samples,
        "device": str(device),
        "split": split_name,
        "encoder_name": actual_encoder,
        "requested_encoder": requested_encoder,
        "actual_encoder": actual_encoder,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "model_name_or_path": encoder_cfg.get("model_name_or_path"),
        "cache_dir": encoder_cfg.get("cache_dir"),
        "encoder_availability": encoder_availability,
        "encoder_runtime_summary": observed_runtime_summary,
        "num_shards": len(shard_summaries),
        "output_dir": str(out_dir),
        "shards": shard_summaries,
        "output_token_shape": shard_summaries[0]["past_tokens_shape"] if shard_summaries else None,
        "elapsed_time_sec": round(time.perf_counter() - start_time, 3),
    }
    summary_path = out_dir / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"EXTRACTION_SUMMARY_WRITTEN = {summary_path}")
    return summary


def extract_tokens_from_config(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    device_name: str | None = None,
    max_samples: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config_path = resolve_path(config_path)
    config = load_yaml(config_path)
    start_time = time.perf_counter()
    dataset_cfg = config["dataset"]
    encoder_cfg = config["encoder"]
    extraction_cfg = config["extraction"]
    fallback_cfg = config.get("fallback", {})

    torch.manual_seed(int(extraction_cfg.get("seed", 42)))
    device = resolve_device(device_name or extraction_cfg.get("device"))
    dataset_name = str(dataset_cfg.get("name", "toy_videos"))
    encoder, actual_encoder, used_fallback, fallback_reason, encoder_availability = _build_encoder(
        encoder_cfg=encoder_cfg,
        fallback_cfg=fallback_cfg,
        device=device,
    )

    if dataset_name == "bair_robot_pushing_small_subset":
        output_root = resolve_path(
            output_dir or extraction_cfg.get("output_root") or extraction_cfg.get("output_dir")
        )
        output_root.mkdir(parents=True, exist_ok=True)
        split_names = [str(split) for split in dataset_cfg.get("split_names", [dataset_cfg.get("split", "train")])]
        split_summaries: dict[str, Any] = {}
        for split_name in split_names:
            effective_max_samples = _resolve_effective_max_samples(
                dataset_cfg, extraction_cfg, max_samples, split=split_name
            )
            dataset = _build_dataset(
                dataset_cfg=dataset_cfg,
                extraction_cfg=extraction_cfg,
                max_samples=effective_max_samples,
                split=split_name,
            )
            split_out_dir = resolve_path(extraction_cfg.get(f"{split_name}_output_dir", output_root / split_name))
            split_summaries[split_name] = _extract_dataset_to_shards(
                config_path=config_path,
                dataset=dataset,
                dataset_cfg=dataset_cfg,
                encoder_cfg=encoder_cfg,
                extraction_cfg=extraction_cfg,
                fallback_cfg=fallback_cfg,
                encoder=encoder,
                actual_encoder=actual_encoder,
                used_fallback=used_fallback,
                fallback_reason=fallback_reason,
                encoder_availability=encoder_availability,
                out_dir=split_out_dir,
                device=device,
                effective_max_samples=effective_max_samples,
                split_name=split_name,
                start_time=start_time,
                overwrite=overwrite,
            )
        overall = {
            "config": str(config_path),
            "dataset_name": dataset_name,
            "splits": split_names,
            "output_root": str(output_root),
            "requested_encoder": str(encoder_cfg.get("name", "dummy_video_encoder")),
            "actual_encoder": actual_encoder,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "encoder_availability": encoder_availability,
            "split_summaries": split_summaries,
            "total_samples": sum(int(summary["dataset_size"]) for summary in split_summaries.values()),
            "total_shards": sum(int(summary["num_shards"]) for summary in split_summaries.values()),
            "elapsed_time_sec": round(time.perf_counter() - start_time, 3),
        }
        summary_path = output_root / "extraction_summary.json"
        summary_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")
        print(json.dumps(overall, indent=2))
        print(f"EXTRACTION_SUMMARY_WRITTEN = {summary_path}")
        return overall

    effective_max_samples = _resolve_effective_max_samples(dataset_cfg, extraction_cfg, max_samples)
    dataset = _build_dataset(
        dataset_cfg=dataset_cfg,
        extraction_cfg=extraction_cfg,
        max_samples=effective_max_samples,
    )
    out_dir = resolve_path(output_dir or extraction_cfg["output_dir"])
    split_name = str(dataset_cfg.get("split", "toy"))
    return _extract_dataset_to_shards(
        config_path=config_path,
        dataset=dataset,
        dataset_cfg=dataset_cfg,
        encoder_cfg=encoder_cfg,
        extraction_cfg=extraction_cfg,
        fallback_cfg=fallback_cfg,
        encoder=encoder,
        actual_encoder=actual_encoder,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        encoder_availability=encoder_availability,
        out_dir=out_dir,
        device=device,
        effective_max_samples=effective_max_samples,
        split_name=split_name,
        start_time=start_time,
        overwrite=overwrite,
    )


def main() -> None:
    args = parse_args()
    extract_tokens_from_config(
        config_path=args.config,
        output_dir=args.output_dir,
        device_name=args.device,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
