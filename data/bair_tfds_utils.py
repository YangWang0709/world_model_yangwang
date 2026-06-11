"""Safe TensorFlow Datasets helpers for BAIR Robot Pushing small."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any


BAIR_TFDS_NAME = "bair_robot_pushing_small"
BAIR_TFDS_VERSION = "2.0.0"


def _versioned_name(tfds_name: str = BAIR_TFDS_NAME, tfds_version: str | None = BAIR_TFDS_VERSION) -> str:
    if not tfds_version:
        return tfds_name
    if ":" in tfds_name:
        return tfds_name
    return f"{tfds_name}:{tfds_version}"


def _import_module(module_name: str) -> tuple[Any | None, str | None]:
    if importlib.util.find_spec(module_name) is None:
        return None, f"{module_name} is not installed"
    try:
        module = __import__(module_name)
    except Exception as exc:  # pragma: no cover - environment-specific import failures
        return None, f"{module_name} import failed: {exc}"
    return module, None


def check_tfds_available() -> dict[str, Any]:
    """Return a capability summary without downloading data."""

    tensorflow, tensorflow_error = _import_module("tensorflow")
    tfds, tfds_error = _import_module("tensorflow_datasets")
    summary: dict[str, Any] = {
        "tensorflow_available": tensorflow is not None,
        "tensorflow_error": tensorflow_error,
        "tensorflow_version": getattr(tensorflow, "__version__", None) if tensorflow is not None else None,
        "tensorflow_datasets_available": tfds is not None,
        "tensorflow_datasets_error": tfds_error,
        "tensorflow_datasets_version": getattr(tfds, "__version__", None) if tfds is not None else None,
        "tfds_builder_available": False,
        "tfds_builder_error": None,
    }
    if tfds is not None:
        try:
            tfds.builder(_versioned_name())
            summary["tfds_builder_available"] = True
        except Exception as exc:  # pragma: no cover - depends on installed TFDS catalog
            summary["tfds_builder_error"] = str(exc)
    return summary


def get_bair_builder(
    data_dir: str,
    tfds_name: str = BAIR_TFDS_NAME,
    tfds_version: str | None = BAIR_TFDS_VERSION,
) -> Any:
    """Create a BAIR TFDS builder or raise an actionable ImportError."""

    tfds, tfds_error = _import_module("tensorflow_datasets")
    if tfds is None:
        raise ImportError(tfds_error or "tensorflow_datasets is not available")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return tfds.builder(_versioned_name(tfds_name, tfds_version), data_dir=str(data_dir))


def _split_summary(builder: Any) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split_name, split_info in builder.info.splits.items():
        splits[str(split_name)] = {
            "num_examples": int(getattr(split_info, "num_examples", 0)),
            "num_shards": int(getattr(split_info, "num_shards", 0)),
        }
    return splits


def get_bair_info(
    data_dir: str,
    tfds_name: str = BAIR_TFDS_NAME,
    tfds_version: str | None = BAIR_TFDS_VERSION,
) -> dict[str, Any]:
    """Return BAIR builder metadata without reading examples."""

    builder = get_bair_builder(data_dir=data_dir, tfds_name=tfds_name, tfds_version=tfds_version)
    return {
        "name": str(builder.info.name),
        "version": str(builder.info.version),
        "data_dir": str(builder.data_dir),
        "features": str(builder.info.features),
        "splits": _split_summary(builder),
        "download_size": str(builder.info.download_size),
        "dataset_size": str(builder.info.dataset_size),
    }


def download_bair_dataset(
    data_dir: str,
    download: bool = True,
    tfds_name: str = BAIR_TFDS_NAME,
    tfds_version: str | None = BAIR_TFDS_VERSION,
) -> dict[str, Any]:
    """Download or inspect the BAIR TFDS dataset."""

    start = time.time()
    builder = get_bair_builder(data_dir=data_dir, tfds_name=tfds_name, tfds_version=tfds_version)
    if download:
        builder.download_and_prepare()
    info = {
        "download": bool(download),
        "elapsed_time_sec": time.time() - start,
        "data_dir": str(builder.data_dir),
        "dataset_info": {
            "name": str(builder.info.name),
            "version": str(builder.info.version),
            "features": str(builder.info.features),
            "splits": _split_summary(builder),
            "download_size": str(builder.info.download_size),
            "dataset_size": str(builder.info.dataset_size),
        },
    }
    return info
