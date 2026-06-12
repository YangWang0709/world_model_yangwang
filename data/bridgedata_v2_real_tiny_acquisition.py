"""Safe acquisition for Step21 BridgeData V2 real tiny samples."""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from data.bridgedata_v2_download_probe import DEFAULT_USER_AGENT, probe_url_size


def acquire_bridgedata_v2_real_tiny_subset(
    config: dict[str, Any],
    probe_summary: dict[str, Any],
) -> dict[str, Any]:
    output_path = Path(config["output"]["acquisition_summary_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_cfg = config["dataset"]
    acquisition_cfg = config["acquisition"]
    max_bytes = int(dataset_cfg["max_download_bytes"])

    if not probe_summary.get("safe_candidate_found"):
        summary = _safe_stop_summary(
            dataset_cfg,
            reason=probe_summary.get("reason") or "No safe official tiny sample candidate found.",
            download_attempted=False,
        )
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    candidate = probe_summary["selected_candidate"]
    if not acquisition_cfg.get("user_approved_tiny_download", False):
        summary = _safe_stop_summary(dataset_cfg, reason="Tiny download is not user-approved.", download_attempted=False)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    second_probe = probe_url_size(candidate["url"], int(config["download_probe"].get("timeout_sec", 30)))
    length = second_probe.get("content_length_bytes")
    if not second_probe.get("size_known") or length is None or length > max_bytes:
        summary = _safe_stop_summary(
            dataset_cfg,
            reason="Second Content-Length check failed or exceeded max_download_bytes.",
            download_attempted=False,
        )
        summary["second_probe"] = second_probe
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    download_root = Path(dataset_cfg["download_root"])
    extract_root = Path(dataset_cfg["extract_root"])
    download_root.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)
    filename = Path(candidate["url"].split("?", 1)[0]).name or "bridgedata_v2_tiny_download.bin"
    target = download_root / filename
    download_result = _download_with_limit(candidate["url"], target, max_bytes)
    if not download_result["download_performed"]:
        summary = _safe_stop_summary(dataset_cfg, reason=download_result["reason"], download_attempted=True)
        summary.update(download_result)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    extraction = _maybe_extract_archive(target, extract_root, int(dataset_cfg["target_limits"]["max_files_to_extract"]))
    summary = {
        "stage": "bridgedata_v2_real_tiny_acquisition",
        "download_attempted": True,
        "download_performed": True,
        "download_bytes": download_result["download_bytes"],
        "download_under_limit": True,
        "extraction_performed": extraction["extraction_performed"],
        "extracted_file_count": extraction["extracted_file_count"],
        "safe_stop": False,
        "reason": None,
        "download_root": str(download_root),
        "extract_root": str(extract_root),
        "download_path": str(target),
        "checksum_sha256": download_result["checksum_sha256"],
        "user_provided_subset_required": False,
        "full_download_performed": False,
        "large_download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "extraction_warnings": extraction["warnings"],
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _safe_stop_summary(dataset_cfg: dict[str, Any], reason: str, download_attempted: bool) -> dict[str, Any]:
    return {
        "stage": "bridgedata_v2_real_tiny_acquisition",
        "download_attempted": download_attempted,
        "download_performed": False,
        "download_bytes": None,
        "download_under_limit": None,
        "extraction_performed": False,
        "extracted_file_count": 0,
        "safe_stop": True,
        "reason": reason,
        "download_root": dataset_cfg["download_root"],
        "extract_root": dataset_cfg["extract_root"],
        "checksum_sha256": None,
        "user_provided_subset_required": True,
        "full_download_performed": False,
        "large_download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
    }


def _download_with_limit(url: str, target: Path, max_bytes: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    total = 0
    sha = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    handle.close()
                    target.unlink(missing_ok=True)
                    return {
                        "download_performed": False,
                        "download_bytes": total,
                        "checksum_sha256": None,
                        "reason": "download exceeded max_download_bytes; partial file deleted",
                    }
                sha.update(chunk)
                handle.write(chunk)
    except Exception as exc:  # pragma: no cover - network dependent
        target.unlink(missing_ok=True)
        return {
            "download_performed": False,
            "download_bytes": total,
            "checksum_sha256": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "download_performed": True,
        "download_bytes": total,
        "checksum_sha256": sha.hexdigest(),
        "reason": None,
    }


def _maybe_extract_archive(path: Path, extract_root: Path, max_files: int) -> dict[str, Any]:
    suffixes = "".join(path.suffixes).lower()
    warnings: list[str] = []
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) > max_files:
                return {"extraction_performed": False, "extracted_file_count": 0, "warnings": ["zip file count exceeds limit"]}
            archive.extractall(extract_root)
            return {"extraction_performed": True, "extracted_file_count": len(names), "warnings": warnings}
    if suffixes.endswith(".tar") or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            if len(members) > max_files:
                return {"extraction_performed": False, "extracted_file_count": 0, "warnings": ["tar file count exceeds limit"]}
            archive.extractall(extract_root)
            return {"extraction_performed": True, "extracted_file_count": len(members), "warnings": warnings}
    return {"extraction_performed": False, "extracted_file_count": 0, "warnings": ["downloaded file is not an archive"]}
