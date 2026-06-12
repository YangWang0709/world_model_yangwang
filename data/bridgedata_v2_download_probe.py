"""Safe official-download probing for Step21 BridgeData V2 tiny validation."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


DATASET_NAME = "BridgeData V2"
DEFAULT_USER_AGENT = "tgpawb-step21-safe-probe/1.0"
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
DATA_HINTS = ("demo", "demos", "scripted", "teleop", "data", "dataset", "bridge")
TINY_HINTS = ("tiny", "sample", "small", "demo_subset", "example")


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self._current_href: str | None = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = {key.lower(): value for key, value in attrs if value is not None}
        href = attrs_dict.get("href")
        if href:
            self._current_href = href
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_href is not None:
            self.links.append({"href": self._current_href, "text": " ".join(self._current_text).strip()})
            self._current_href = None
            self._current_text = []


def probe_url_size(url: str, timeout_sec: int = 30) -> dict[str, Any]:
    """Probe URL metadata with HEAD only. No file body is downloaded."""

    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": DEFAULT_USER_AGENT})
    result: dict[str, Any] = {
        "url": url,
        "method": "HEAD",
        "ok": False,
        "status": None,
        "content_length_bytes": None,
        "size_known": False,
        "content_type": None,
        "final_url": url,
        "error": None,
    }
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            length = response.headers.get("Content-Length")
            result.update(
                {
                    "ok": True,
                    "status": getattr(response, "status", None),
                    "content_length_bytes": int(length) if length and length.isdigit() else None,
                    "size_known": bool(length and length.isdigit()),
                    "content_type": response.headers.get("Content-Type"),
                    "final_url": response.geturl(),
                }
            )
    except Exception as exc:  # pragma: no cover - exercised by network environments
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def classify_download_candidate(candidate: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    url = candidate.get("url", "")
    text = (candidate.get("name") or candidate.get("link_text") or "").lower()
    lower_url = url.lower()
    probe = candidate.get("probe", {})
    content_length = probe.get("content_length_bytes")
    size_known = bool(probe.get("size_known"))

    candidate_kind = "official_metadata"
    if "droid" in lower_url or "droid" in text:
        candidate_kind = "droid"
    elif "open_x" in lower_url or "open-x" in lower_url or "open x" in text:
        candidate_kind = "open_x_embodiment"
    elif any(hint in lower_url or hint in text for hint in TINY_HINTS) and lower_url.endswith(ARCHIVE_SUFFIXES):
        candidate_kind = "official_tiny_sample"
    elif lower_url.endswith(ARCHIVE_SUFFIXES) and any(hint in lower_url or hint in text for hint in DATA_HINTS):
        if "scripted" in lower_url or "scripted" in text:
            candidate_kind = "full_scripted_zip"
        elif "demo" in lower_url or "teleop" in lower_url or "demo" in text or "teleop" in text:
            candidate_kind = "full_teleop_zip"
        else:
            candidate_kind = "official_single_small_zip"
    elif lower_url.endswith(ARCHIVE_SUFFIXES):
        candidate_kind = "unknown_size_file"

    safe_under_limit = bool(size_known and content_length is not None and content_length <= max_bytes)
    downloadable_data_kind = candidate_kind in {
        "official_tiny_sample",
        "official_single_small_zip",
        "official_demo_subset_if_size_known",
    }
    selected_safe = bool(downloadable_data_kind and safe_under_limit)
    if not size_known:
        reason = "size unknown"
    elif content_length is not None and content_length > max_bytes:
        reason = "larger than max_download_bytes"
    elif not downloadable_data_kind:
        reason = "not a tiny data archive candidate"
    else:
        reason = None
    classified = dict(candidate)
    classified.update(
        {
            "candidate_kind": candidate_kind,
            "content_length_bytes": content_length,
            "size_known": size_known,
            "safe_under_limit": safe_under_limit,
            "selected_safe": selected_safe,
            "unsafe_reason": reason,
        }
    )
    return classified


def probe_bridgedata_official_download_options(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    probe_cfg = config.get("download_probe", {})
    max_bytes = int(dataset_cfg["max_download_bytes"])
    timeout = int(probe_cfg.get("timeout_sec", 30))
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []

    for source in dataset_cfg.get("official_sources", []):
        source_url = source["url"]
        page = _fetch_text_page(source_url, timeout)
        if page["ok"]:
            links = _extract_links(page["text"], source_url)
            for link in links:
                if _looks_like_candidate(link["url"], link["text"]):
                    probe = probe_url_size(link["url"], timeout)
                    candidates.append(
                        {
                            "name": link["text"] or Path(urllib.parse.urlparse(link["url"]).path).name or link["url"],
                            "url": link["url"],
                            "url_source": source["kind"],
                            "source_page": source_url,
                            "link_text": link["text"],
                            "probe": probe,
                        }
                    )
        else:
            warnings.append(f"failed to read official source {source_url}: {page['error']}")

    classified = [classify_download_candidate(candidate, max_bytes) for candidate in candidates]
    safe_candidates = [candidate for candidate in classified if candidate["selected_safe"]]
    selected = safe_candidates[0] if safe_candidates else None
    safe_stop = selected is None
    reason = None if selected else "No official <=1GB tiny sample found or file size unknown."
    return {
        "stage": "bridgedata_v2_download_probe",
        "dataset_name": DATASET_NAME,
        "official_sources_checked": True,
        "official_sources": dataset_cfg.get("official_sources", []),
        "download_candidates": classified,
        "safe_candidate_found": selected is not None,
        "selected_candidate": selected,
        "safe_stop": safe_stop,
        "reason": reason,
        "warnings": warnings,
        "max_download_bytes": max_bytes,
        "no_large_download_performed": True,
        "download_performed": False,
    }


def write_probe_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def _fetch_text_page(url: str, timeout_sec: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read(2_000_000)
            content_type = response.headers.get("Content-Type", "")
            return {
                "ok": True,
                "text": raw.decode("utf-8", "replace"),
                "content_type": content_type,
                "error": None,
            }
    except Exception as exc:  # pragma: no cover - exercised by network environments
        return {"ok": False, "text": "", "content_type": None, "error": f"{type(exc).__name__}: {exc}"}


def _extract_links(html: str, base_url: str) -> list[dict[str, str]]:
    parser = _LinkParser()
    parser.feed(html)
    links = []
    for link in parser.links:
        href = link["href"]
        if href.startswith("#") or href.startswith("mailto:"):
            continue
        links.append({"url": urllib.parse.urljoin(base_url, href), "text": re.sub(r"\s+", " ", link["text"]).strip()})
    return links


def _looks_like_candidate(url: str, text: str) -> bool:
    lower = f"{url} {text}".lower()
    if any(term in lower for term in ("droid", "open-x", "open_x", "open x")):
        return False
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    data_directory = "bridge_release/data" in lower or "/datasets/" in path and "/data" in path
    archive = path.endswith(ARCHIVE_SUFFIXES)
    explicit_download = any(term in lower for term in ("download", "demo", "scripted", "teleop", "sample", "tiny"))
    return bool(data_directory or archive or explicit_download)
