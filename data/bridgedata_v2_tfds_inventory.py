"""Inventory official BridgeData V2 TFDS/RLDS directory listings."""

from __future__ import annotations

import fnmatch
import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


APACHE_ROW_RE = re.compile(
    r'<a\s+href="(?P<href>[^"]+)">(?P<label>[^<]+)</a>\s+'
    r'(?P<last_modified>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s+'
    r'(?P<size>\S+)',
    re.IGNORECASE,
)
APACHE_TABLE_ROW_RE = re.compile(
    r'<tr>.*?<a\s+href="(?P<href>[^"]+)">(?P<label>.*?)</a>.*?'
    r'<td[^>]*>\s*(?P<last_modified>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*</td>.*?'
    r'<td[^>]*>\s*(?P<size>[^<]+?)\s*</td>',
    re.IGNORECASE | re.DOTALL,
)


def parse_size_to_bytes(token: str | None) -> int | None:
    if token is None or token == "-":
        return None
    text = token.strip()
    match = re.fullmatch(r"(?P<num>\d+(?:\.\d+)?)(?P<unit>[KMGTP]?)", text, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group("num"))
    unit = match.group("unit").upper()
    multiplier = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
    }[unit]
    return int(value * multiplier)


def parse_tfds_index_html(
    html_text: str,
    base_url: str,
    *,
    metadata_files: list[str],
    optional_metadata_patterns: list[str],
    shard_name_prefix: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    matches = list(APACHE_TABLE_ROW_RE.finditer(html_text)) or list(APACHE_ROW_RE.finditer(html_text))
    for match in matches:
        href = html.unescape(match.group("href"))
        if href in {"../", "./"} or href.endswith("/"):
            continue
        name = urllib.parse.unquote(Path(href).name)
        size_token = html.unescape(match.group("size"))
        size_bytes = parse_size_to_bytes(size_token)
        url = urllib.parse.urljoin(base_url, href)
        entry = {
            "name": name,
            "url": url,
            "last_modified": match.group("last_modified"),
            "size_text": size_token,
            "size_bytes": size_bytes,
            "size_known": size_bytes is not None,
            "is_metadata": name in metadata_files
            or any(fnmatch.fnmatch(name, pattern) for pattern in optional_metadata_patterns),
            "is_train_shard": name.startswith(shard_name_prefix),
            "is_raw_zip": name.endswith(".zip") or "demos_" in name or "scripted_" in name,
        }
        entries.append(entry)
    metadata = [entry for entry in entries if entry["is_metadata"]]
    train_shards = [entry for entry in entries if entry["is_train_shard"] and not entry["is_raw_zip"]]
    return {
        "entries": entries,
        "metadata_files": metadata,
        "train_shards": train_shards,
        "raw_zip_candidates": [entry for entry in entries if entry["is_raw_zip"]],
    }


def select_train_shards(
    train_shards: list[dict[str, Any]],
    *,
    preferred_shard_count: int,
    max_shard_count: int,
    max_download_bytes: int,
) -> tuple[list[dict[str, Any]], int, bool]:
    selected: list[dict[str, Any]] = []
    total = 0
    target = min(preferred_shard_count, max_shard_count)
    for shard in train_shards:
        size = shard.get("size_bytes")
        if size is None:
            continue
        if total + int(size) > max_download_bytes:
            continue
        selected.append(shard)
        total += int(size)
        if len(selected) >= target:
            break
    return selected, total, bool(selected and total <= max_download_bytes)


def inventory_bridgedata_v2_tfds(config: dict[str, Any], *, html_text: str | None = None) -> dict[str, Any]:
    dataset = config["dataset"]
    policy = config["download_policy"]
    base_url = dataset["official_tfds_base_url"]
    try:
        if html_text is None:
            with urllib.request.urlopen(base_url, timeout=30) as response:
                html_text = response.read().decode("utf-8", "replace")
        parsed = parse_tfds_index_html(
            html_text,
            base_url,
            metadata_files=list(dataset.get("metadata_files", [])),
            optional_metadata_patterns=list(dataset.get("optional_metadata_patterns", [])),
            shard_name_prefix=policy["shard_name_prefix"],
        )
        selected, total, under_limit = select_train_shards(
            parsed["train_shards"],
            preferred_shard_count=int(policy["preferred_tfds_train_shards"]),
            max_shard_count=int(policy["max_tfds_train_shards"]),
            max_download_bytes=int(policy["max_download_bytes"]),
        )
        summary = {
            "stage": "bridgedata_v2_tfds_inventory",
            "official_tfds_base_url": base_url,
            "safe_stop": False,
            "reason": None,
            "metadata_files": parsed["metadata_files"],
            "train_shards": parsed["train_shards"],
            "num_train_shards_found": len(parsed["train_shards"]),
            "preferred_shard_count": int(policy["preferred_tfds_train_shards"]),
            "max_shard_count": int(policy["max_tfds_train_shards"]),
            "selected_shards": selected,
            "selected_total_bytes": total,
            "selected_under_limit": under_limit,
            "raw_zip_candidates_ignored": len(parsed["raw_zip_candidates"]),
            "no_raw_zip": True,
        }
        if not selected:
            summary.update({"safe_stop": True, "reason": "No known-size train shard fits the mini-shard download policy."})
        return summary
    except Exception as exc:
        return {
            "stage": "bridgedata_v2_tfds_inventory",
            "official_tfds_base_url": base_url,
            "safe_stop": True,
            "reason": f"Could not read official TFDS listing: {exc}",
            "metadata_files": [],
            "train_shards": [],
            "num_train_shards_found": 0,
            "preferred_shard_count": int(policy["preferred_tfds_train_shards"]),
            "max_shard_count": int(policy["max_tfds_train_shards"]),
            "selected_shards": [],
            "selected_total_bytes": 0,
            "selected_under_limit": False,
            "raw_zip_candidates_ignored": 0,
            "no_raw_zip": True,
        }


def write_inventory_summary(summary: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
