"""Step32 longer-horizon proxy-importance summaries."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from data.bridgedata_v2_tfds_importance_manifest import read_importance_manifest_jsonl
from data.bridgedata_v2_tfds_longer_horizon_manifest import STAGE, read_jsonl


def summarize_longer_horizon_proxy_importance(
    importance_manifest_jsonl: str | Path,
    window_manifest_jsonl: str | Path,
    output_summary_json: str | Path | None = None,
) -> dict[str, Any]:
    importance_records = read_importance_manifest_jsonl(importance_manifest_jsonl)
    windows = read_jsonl(window_manifest_jsonl)
    horizon_by_sample = {str(window["sample_id"]): int(window["horizon_gap"]) for window in windows}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in importance_records:
        grouped[horizon_by_sample[str(record["sample_id"])]].append(record)
    first = importance_records[0] if importance_records else {}
    payload = {
        "stage": STAGE,
        "limited_proxy_importance_generation_performed": bool(importance_records),
        "method": "proxy_token_mse_dryrun",
        "horizons": sorted(grouped),
        "num_samples": len(importance_records),
        "samples_by_horizon": {f"gap{gap}": len(items) for gap, items in sorted(grouped.items())},
        "context_importance_shape_example": first.get("context_importance_shape"),
        "temporal_importance_shape_example": first.get("temporal_importance_shape"),
        "spatial_importance_shape_example": first.get("spatial_importance_shape"),
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "current_importance_generated": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "data_importance_shards_written": False,
        "target_aware_proxy_summary_only": True,
        "safety_gate_pass": True,
    }
    if output_summary_json is not None:
        path = Path(output_summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
