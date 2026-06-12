"""Acquire at most one extra BridgeData V2 TFDS train shard for Step33B."""

from __future__ import annotations

import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


HeadFunc = Callable[[str], int | None]
OpenFunc = Callable[[str], Any]


STAGE = "bridgedata_v2_tfds_second_shard_acquisition_step33b"


def expected_shard_filename(pattern: str, index: int) -> str:
    name = str(pattern).format(index=int(index))
    if not name.startswith("bridge_dataset-train.tfrecord-"):
        raise ValueError(f"unexpected TFDS shard filename: {name}")
    if ".zip" in name.lower():
        raise ValueError(f"raw zip names are forbidden: {name}")
    return name


def acquire_second_shard(
    *,
    dataset_root: str | Path,
    base_url: str,
    preferred_indices: list[int],
    expected_filename_pattern: str,
    max_new_shards_to_download: int = 1,
    warning_size_mb: int = 256,
    hard_cap_size_mb: int = 1024,
    head_func: HeadFunc | None = None,
    open_func: OpenFunc | None = None,
) -> dict[str, Any]:
    """Reuse an existing shard or download one known-size `.tfrecord-*` shard."""

    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    if int(max_new_shards_to_download) != 1:
        return _safe_stop(root, "Step33B only allows max_new_shards_to_download=1")
    head = head_func or _head_content_length
    attempted: list[dict[str, Any]] = []
    for index in [int(item) for item in preferred_indices]:
        try:
            filename = expected_shard_filename(expected_filename_pattern, index)
        except ValueError as exc:
            attempted.append({"index": index, "safe_stop_reason": str(exc)})
            return _safe_stop(root, str(exc), attempted)
        destination = root / filename
        url = urllib.parse.urljoin(str(base_url).rstrip("/") + "/", filename)
        if _is_forbidden_url(url):
            attempted.append({"index": index, "filename": filename, "url": url, "safe_stop_reason": "forbidden URL"})
            continue
        if destination.exists():
            size = int(destination.stat().st_size)
            return {
                "stage": STAGE,
                "safe_stop": False,
                "reason": None,
                "second_shard_available": True,
                "selected_shard_index": index,
                "selected_shard_filename": filename,
                "selected_shard_path": str(destination),
                "selected_shard_url": url,
                "selected_shard_size_bytes": size,
                "selected_shard_size_mb": size / (1024**2),
                "downloaded_new_shard_count": 0,
                "download_performed": False,
                "reused_existing_shard": True,
                "attempted_shards": attempted,
                "raw_zip_downloaded": False,
                "full_tfds_downloaded": False,
                "droid_downloaded": False,
                "model_download_performed": False,
                "safety_gate_pass": True,
            }
        content_length = head(url)
        attempted.append(
            {
                "index": index,
                "filename": filename,
                "url": url,
                "content_length": content_length,
                "content_length_known": content_length is not None,
            }
        )
        if content_length is None:
            return _safe_stop(root, "Content-Length is unknown for candidate shard", attempted)
        hard_cap = int(hard_cap_size_mb) * 1024 * 1024
        if int(content_length) > hard_cap:
            return _safe_stop(root, "candidate shard exceeds hard cap", attempted)
        warning = int(warning_size_mb) * 1024 * 1024
        _download_file(url, destination, int(content_length), hard_cap, open_func=open_func)
        actual = int(destination.stat().st_size)
        if actual != int(content_length):
            destination.unlink(missing_ok=True)
            return _safe_stop(root, "downloaded size mismatch for candidate shard", attempted)
        return {
            "stage": STAGE,
            "safe_stop": False,
            "reason": None,
            "second_shard_available": True,
            "selected_shard_index": index,
            "selected_shard_filename": filename,
            "selected_shard_path": str(destination),
            "selected_shard_url": url,
            "selected_shard_size_bytes": actual,
            "selected_shard_size_mb": actual / (1024**2),
            "warning_threshold_exceeded": bool(actual > warning),
            "downloaded_new_shard_count": 1,
            "download_performed": True,
            "reused_existing_shard": False,
            "attempted_shards": attempted,
            "raw_zip_downloaded": False,
            "full_tfds_downloaded": False,
            "droid_downloaded": False,
            "model_download_performed": False,
            "safety_gate_pass": True,
        }
    return _safe_stop(root, "no candidate second shard could be safely acquired", attempted)


def write_acquisition_summary(summary: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _head_content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        value = response.headers.get("Content-Length")
    return int(value) if value and value.isdigit() else None


def _download_file(
    url: str,
    destination: Path,
    expected_bytes: int,
    hard_cap_bytes: int,
    *,
    open_func: OpenFunc | None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    opener = open_func or (lambda item_url: urllib.request.urlopen(item_url, timeout=60))
    written = 0
    try:
        with opener(url) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > expected_bytes or written > hard_cap_bytes:
                    raise RuntimeError("download exceeded expected size or hard cap")
                handle.write(chunk)
        shutil.move(str(partial), str(destination))
    except Exception:
        partial.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise


def _is_forbidden_url(url: str) -> bool:
    lower = url.lower()
    return lower.endswith(".zip") or "demos_" in lower or "scripted_" in lower


def _safe_stop(
    root: Path,
    reason: str,
    attempted: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "safe_stop": True,
        "reason": reason,
        "second_shard_available": False,
        "selected_shard_index": None,
        "selected_shard_filename": None,
        "selected_shard_path": None,
        "selected_shard_size_bytes": 0,
        "downloaded_new_shard_count": 0,
        "download_performed": False,
        "reused_existing_shard": False,
        "attempted_shards": attempted or [],
        "dataset_root": str(root),
        "raw_zip_downloaded": False,
        "full_tfds_downloaded": False,
        "droid_downloaded": False,
        "model_download_performed": False,
        "safety_gate_pass": True,
    }
