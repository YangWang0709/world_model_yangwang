"""Build Step32 longer-horizon windows from the existing resolved TFDS manifest."""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data.bridgedata_v2_tfds_longer_horizon_manifest import (
    CURRENT_LEN,
    CONTEXT_LEN,
    FUTURE_LEN,
    STAGE,
    read_jsonl,
    validate_longer_horizon_window,
    write_jsonl,
)


def build_longer_horizon_windows_from_manifest(
    resolved_manifest_jsonl: str | Path,
    output_manifest_jsonl: str | Path,
    output_summary_json: str | Path,
    *,
    horizon_gaps: list[int],
    required_horizon_gaps: list[int],
    optional_horizon_gaps: list[int],
    target_windows_per_horizon: int = 64,
    min_windows_per_horizon: int = 32,
    max_windows_per_trajectory_soft_cap: int = 16,
    deterministic_seed: int = 42,
) -> dict[str, Any]:
    base_windows = read_jsonl(resolved_manifest_jsonl)
    selected: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for gap in horizon_gaps:
        candidates = _candidate_windows(base_windows, int(gap))
        chosen = _select_diverse_windows(
            candidates,
            target_count=target_windows_per_horizon,
            soft_cap=max_windows_per_trajectory_soft_cap,
            seed=deterministic_seed + int(gap),
        )
        safe_skipped = len(candidates) < min_windows_per_horizon
        if safe_skipped:
            chosen = []
        selected.extend(chosen)
        summaries[f"gap{gap}"] = _horizon_summary(gap, candidates, chosen, safe_skipped, min_windows_per_horizon)

    required_missing = [
        int(gap)
        for gap in required_horizon_gaps
        if bool(summaries.get(f"gap{gap}", {}).get("safe_skipped", True))
    ]
    safe_stop = bool(required_missing)
    for record in selected:
        validate_longer_horizon_window(record)
    write_jsonl(selected, output_manifest_jsonl)
    summary = {
        "stage": STAGE,
        "safe_stop": safe_stop,
        "reason": f"required horizons safe-skipped: {required_missing}" if required_missing else None,
        "longer_horizon_window_builder_performed": True,
        "horizon_gaps": [int(gap) for gap in horizon_gaps],
        "required_horizon_gaps": [int(gap) for gap in required_horizon_gaps],
        "optional_horizon_gaps": [int(gap) for gap in optional_horizon_gaps],
        "horizons": summaries,
        "horizons_completed": [
            int(gap) for gap in horizon_gaps if not bool(summaries.get(f"gap{gap}", {}).get("safe_skipped"))
        ],
        "required_horizons_completed": [
            int(gap)
            for gap in required_horizon_gaps
            if not bool(summaries.get(f"gap{gap}", {}).get("safe_skipped"))
        ],
        "optional_horizons_completed": [
            int(gap)
            for gap in optional_horizon_gaps
            if not bool(summaries.get(f"gap{gap}", {}).get("safe_skipped"))
        ],
        "num_selected_windows_total": len(selected),
        "num_selected_trajectories_total": len({str(record["trajectory_id"]) for record in selected}),
        "action_language_goal_as_metadata_only": True,
        "new_tfds_shard_downloaded": False,
        "raw_zip_downloaded": False,
        "full_tfds_downloaded": False,
        "download_performed": False,
        "safety_gate_pass": True,
    }
    _write_json(output_summary_json, summary)
    return summary


def build_window_from_gap0_record(record: dict[str, Any], *, horizon_gap: int) -> dict[str, Any]:
    start = int(record["context_frame_indices"][0])
    context = list(range(start, start + CONTEXT_LEN))
    current_start = start + CONTEXT_LEN
    current = list(range(current_start, current_start + CURRENT_LEN))
    gap_start = current_start + CURRENT_LEN
    gap_indices = list(range(gap_start, gap_start + int(horizon_gap)))
    future_start = gap_start + int(horizon_gap)
    future = list(range(future_start, future_start + FUTURE_LEN))
    metadata = copy.deepcopy(record.get("metadata") or {})
    metadata.update(
        {
            "step32_longer_horizon": True,
            "source_gap0_sample_id": record.get("sample_id"),
            "source_context_start": start,
            "horizon_gap": int(horizon_gap),
            "total_span": CONTEXT_LEN + CURRENT_LEN + int(horizon_gap) + FUTURE_LEN,
            "gap_frame_indices": gap_indices,
            "use_action_as_input": False,
            "use_language_as_input": False,
            "use_goal_image_as_input": False,
        }
    )
    out = copy.deepcopy(record)
    out.update(
        {
            "sample_id": f"gap{int(horizon_gap)}_{record['trajectory_id']}_window_{start:06d}",
            "context_frame_indices": context,
            "current_frame_indices": current,
            "gap_frame_indices": gap_indices,
            "future_frame_indices": future,
            "horizon_gap": int(horizon_gap),
            "total_span": CONTEXT_LEN + CURRENT_LEN + int(horizon_gap) + FUTURE_LEN,
            "metadata": metadata,
        }
    )
    validate_longer_horizon_window(out)
    return out


def _candidate_windows(base_windows: list[dict[str, Any]], gap: int) -> list[dict[str, Any]]:
    starts_by_traj = defaultdict(set)
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in base_windows:
        trajectory_id = str(record["trajectory_id"])
        start = int(record["context_frame_indices"][0])
        starts_by_traj[trajectory_id].add(start)
        by_key[(trajectory_id, start)] = record

    candidates: list[dict[str, Any]] = []
    for (trajectory_id, start), record in sorted(by_key.items()):
        if (start + gap) in starts_by_traj[trajectory_id]:
            candidates.append(build_window_from_gap0_record(record, horizon_gap=gap))
    return candidates


def _select_diverse_windows(
    candidates: list[dict[str, Any]],
    *,
    target_count: int,
    soft_cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    del seed  # The ordering is deterministic by trajectory and start for reproducibility.
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        grouped[str(record["trajectory_id"])].append(record)
    for records in grouped.values():
        records.sort(key=lambda item: int(item["context_frame_indices"][0]))

    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    while len(selected) < target_count:
        progressed = False
        for trajectory_id in sorted(grouped):
            if len(selected) >= target_count:
                break
            if counts[trajectory_id] >= soft_cap:
                continue
            records = grouped[trajectory_id]
            if counts[trajectory_id] < len(records):
                selected.append(records[counts[trajectory_id]])
                counts[trajectory_id] += 1
                progressed = True
        if not progressed:
            break

    if len(selected) < min(target_count, len(candidates)):
        already = {str(record["sample_id"]) for record in selected}
        for record in sorted(candidates, key=lambda item: (str(item["trajectory_id"]), int(item["context_frame_indices"][0]))):
            if len(selected) >= target_count:
                break
            if str(record["sample_id"]) not in already:
                selected.append(record)
                already.add(str(record["sample_id"]))
    return selected[: min(target_count, len(candidates))]


def _horizon_summary(
    gap: int,
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    safe_skipped: bool,
    min_windows_per_horizon: int,
) -> dict[str, Any]:
    selected_counts = Counter(str(record["trajectory_id"]) for record in selected)
    return {
        "horizon_gap": int(gap),
        "available_windows": len(candidates),
        "selected_windows": len(selected),
        "num_trajectories": len({str(record["trajectory_id"]) for record in selected}),
        "available_trajectories": len({str(record["trajectory_id"]) for record in candidates}),
        "trajectory_window_counts": dict(sorted(selected_counts.items())),
        "safe_skipped": bool(safe_skipped),
        "safe_skip_reason": f"available_windows < min_windows_per_horizon ({min_windows_per_horizon})"
        if safe_skipped
        else None,
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
