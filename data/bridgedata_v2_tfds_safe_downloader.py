"""Safe downloader for tiny official BridgeData V2 TFDS/RLDS mini-shards."""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Callable


HeadFunc = Callable[[str], int | None]
OpenFunc = Callable[[str], Any]


def is_forbidden_raw_url(url: str) -> bool:
    lower = url.lower()
    return lower.endswith(".zip") or "demos_" in lower or "scripted_" in lower


def head_content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        value = response.headers.get("Content-Length")
    return int(value) if value and value.isdigit() else None


def validate_download_item(url: str, content_length: int | None, *, max_download_bytes: int) -> tuple[bool, str | None]:
    if is_forbidden_raw_url(url):
        return False, "raw zip/demo/scripted URL is forbidden"
    if content_length is None:
        return False, "Content-Length is unknown"
    if content_length > max_download_bytes:
        return False, "file exceeds max_download_bytes"
    return True, None


def plan_download_items(inventory: dict[str, Any], *, include_metadata: bool = True) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if include_metadata:
        items.extend({**entry, "kind": "metadata"} for entry in inventory.get("metadata_files", []))
    items.extend({**entry, "kind": "train_shard"} for entry in inventory.get("selected_shards", []))
    return items


def download_tfds_mini_shards(
    config: dict[str, Any],
    inventory: dict[str, Any],
    *,
    head_func: HeadFunc = head_content_length,
    open_func: OpenFunc | None = None,
) -> dict[str, Any]:
    policy = config["download_policy"]
    paths = config["paths"]
    max_bytes = int(policy["max_download_bytes"])
    dataset_root = Path(paths["tfds_dataset_root"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    if inventory.get("safe_stop"):
        return _safe_stop(paths, inventory.get("reason") or "inventory safe-stopped")
    if not inventory.get("selected_under_limit"):
        return _safe_stop(paths, "selected shards are not under the configured download limit")

    downloaded_files: list[dict[str, Any]] = []
    skipped_files: list[dict[str, Any]] = []
    total_bytes = 0
    download_performed = False
    num_shards_available = 0
    for item in plan_download_items(inventory, include_metadata=bool(policy.get("allow_metadata_download", True))):
        url = item["url"]
        if is_forbidden_raw_url(url):
            skipped_files.append({"name": item["name"], "url": url, "reason": "forbidden raw URL"})
            continue
        content_length = head_func(url)
        ok, reason = validate_download_item(url, content_length, max_download_bytes=max_bytes)
        if not ok:
            skipped_files.append({"name": item["name"], "url": url, "reason": reason})
            if item.get("kind") == "train_shard":
                return _safe_stop(paths, f"Required train shard rejected: {reason}", skipped_files=skipped_files)
            continue
        assert content_length is not None
        if total_bytes + content_length > max_bytes:
            return _safe_stop(paths, "download plan would exceed max_download_bytes", skipped_files=skipped_files)
        destination = dataset_root / item["name"]
        if destination.exists() and destination.stat().st_size == content_length:
            downloaded_files.append(
                {
                    "name": item["name"],
                    "path": str(destination),
                    "url": url,
                    "bytes": content_length,
                    "kind": item.get("kind"),
                    "already_present": True,
                }
            )
            total_bytes += content_length
            if item.get("kind") == "train_shard":
                num_shards_available += 1
            continue
        try:
            _download_file(url, destination, content_length, max_bytes - total_bytes, open_func=open_func)
        except Exception as exc:
            return _safe_stop(paths, f"download failed for {item['name']}: {exc}", skipped_files=skipped_files)
        actual = destination.stat().st_size
        if actual != content_length:
            destination.unlink(missing_ok=True)
            return _safe_stop(paths, f"downloaded size mismatch for {item['name']}", skipped_files=skipped_files)
        total_bytes += actual
        download_performed = True
        if item.get("kind") == "train_shard":
            num_shards_available += 1
        downloaded_files.append(
            {
                "name": item["name"],
                "path": str(destination),
                "url": url,
                "bytes": actual,
                "kind": item.get("kind"),
                "already_present": False,
            }
        )

    if num_shards_available <= 0:
        return _safe_stop(paths, "No train shards were safely downloaded or already present.", skipped_files=skipped_files)
    metadata_available = any(item.get("kind") == "metadata" for item in downloaded_files)
    return {
        "stage": "bridgedata_v2_tfds_mini_download",
        "download_performed": bool(download_performed or num_shards_available > 0),
        "metadata_downloaded": metadata_available,
        "num_shards_downloaded": num_shards_available,
        "download_bytes": total_bytes,
        "under_limit": total_bytes <= max_bytes,
        "raw_zip_downloaded": False,
        "full_tfds_downloaded": False,
        "droid_downloaded": False,
        "download_root": paths["download_root"],
        "tfds_dataset_root": paths["tfds_dataset_root"],
        "downloaded_files": downloaded_files,
        "skipped_files": skipped_files,
        "safe_stop": False,
        "reason": None,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
    }


def write_download_summary(summary: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _download_file(url: str, destination: Path, expected_bytes: int, remaining_budget: int, *, open_func: OpenFunc | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
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
                if written > expected_bytes or written > remaining_budget:
                    raise RuntimeError("download exceeded expected size or remaining budget")
                handle.write(chunk)
        shutil.move(str(partial), str(destination))
    except Exception:
        partial.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise


def _safe_stop(paths: dict[str, Any], reason: str, *, skipped_files: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "stage": "bridgedata_v2_tfds_mini_download",
        "download_performed": False,
        "metadata_downloaded": False,
        "num_shards_downloaded": 0,
        "download_bytes": 0,
        "under_limit": True,
        "raw_zip_downloaded": False,
        "full_tfds_downloaded": False,
        "droid_downloaded": False,
        "download_root": paths["download_root"],
        "tfds_dataset_root": paths["tfds_dataset_root"],
        "downloaded_files": [],
        "skipped_files": skipped_files or [],
        "safe_stop": True,
        "reason": reason,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
    }
