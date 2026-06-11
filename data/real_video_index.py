"""Build metadata indexes for the Step 9A real-video minimal dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .real_video_transforms import ensure_video_tensor


FRAME_EXTENSIONS = {".png", ".jpg", ".jpeg"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class IndexBuildSummary:
    metadata_path: str
    num_samples: int
    source_counts: dict[str, int]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_path": self.metadata_path,
            "num_samples": self.num_samples,
            "source_counts": dict(self.source_counts),
            "warnings": list(self.warnings),
        }


def read_metadata_jsonl(path: str | Path) -> list[dict[str, Any]]:
    metadata_path = Path(path)
    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {metadata_path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Metadata line must be a JSON object at {metadata_path}:{line_number}")
            records.append(record)
    return records


def write_metadata_jsonl(path: str | Path, records: list[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return output_path


def _metadata_path_for(source_path: Path, metadata_path: Path) -> str:
    base_dir = metadata_path.parent.resolve()
    resolved = source_path.resolve()
    try:
        return resolved.relative_to(base_dir).as_posix()
    except ValueError:
        return str(resolved)


def _inspect_pt_clip(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - exact torch exception varies by version
        raise ValueError(f"Could not load .pt clip {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f".pt clip payload must be a dict: {path}")
    if "video" not in payload:
        raise KeyError(f".pt clip payload missing 'video': {path}")
    video = ensure_video_tensor(payload["video"])
    return {
        "num_frames": int(video.shape[0]),
        "height": int(video.shape[2]),
        "width": int(video.shape[3]),
        "fps": int(payload.get("fps", 5)),
    }


def _inspect_frame_folder(path: Path) -> dict[str, Any]:
    frame_paths = sorted(child for child in path.iterdir() if child.suffix.lower() in FRAME_EXTENSIONS)
    if not frame_paths:
        raise ValueError(f"Frame folder has no supported image frames: {path}")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("Frame folders require PIL/Pillow for image size inspection") from exc

    with Image.open(frame_paths[0]) as image:
        width, height = image.convert("RGB").size
    return {
        "num_frames": len(frame_paths),
        "height": int(height),
        "width": int(width),
        "fps": 5,
    }


def _try_inspect_video_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        import imageio.v3 as iio  # type: ignore
    except ImportError:
        return None, f"video_file unsupported without imageio backend: {path}"

    try:
        props = iio.improps(path)
        shape = tuple(props.shape)
    except Exception as exc:  # pragma: no cover - optional backend path
        return None, f"video_file unsupported for {path}: {exc}"

    if len(shape) < 3:
        return None, f"video_file unsupported because shape is unknown for {path}"
    if len(shape) == 4:
        num_frames, height, width = int(shape[0]), int(shape[1]), int(shape[2])
    else:
        num_frames, height, width = 0, int(shape[0]), int(shape[1])
    return {
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "fps": 5,
    }, None


def _iter_frame_folders(input_dir: Path) -> list[Path]:
    folders: list[Path] = []
    for folder in [input_dir, *sorted(child for child in input_dir.rglob("*") if child.is_dir())]:
        if any(child.suffix.lower() in FRAME_EXTENSIONS for child in folder.iterdir() if child.is_file()):
            folders.append(folder)
    return folders


def discover_real_video_sources(input_dir: str | Path) -> tuple[list[tuple[str, Path]], list[str]]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {root}")

    sources: list[tuple[str, Path]] = []
    warnings: list[str] = []
    sources.extend(("pt_clip", path) for path in sorted(root.rglob("*.pt")) if path.is_file())
    sources.extend(("frame_folder", path) for path in _iter_frame_folders(root))

    for video_path in sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS):
        info, warning = _try_inspect_video_file(video_path)
        if info is None:
            warnings.append(warning or f"video_file unsupported: {video_path}")
        else:
            sources.append(("video_file", video_path))

    return sorted(sources, key=lambda item: str(item[1])), warnings


def build_real_video_index(
    input_dir: str | Path,
    output_metadata: str | Path,
    source: str = "real_video_minimal",
    max_samples: int | None = None,
    min_frames: int = 8,
    task_text: str = "predict future visual dynamics",
    split: str = "real_minimal",
) -> dict[str, Any]:
    """Scan input media and write a unified metadata.jsonl file."""

    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    if min_frames <= 0:
        raise ValueError("min_frames must be positive")

    metadata_path = Path(output_metadata)
    discovered, warnings = discover_real_video_sources(input_dir)
    records: list[dict[str, Any]] = []
    source_counts = {"pt_clip": 0, "frame_folder": 0, "video_file": 0}

    for source_type, path in discovered:
        if max_samples is not None and len(records) >= max_samples:
            break
        if source_type == "pt_clip":
            info = _inspect_pt_clip(path)
        elif source_type == "frame_folder":
            info = _inspect_frame_folder(path)
        elif source_type == "video_file":
            info, warning = _try_inspect_video_file(path)
            if info is None:
                warnings.append(warning or f"video_file unsupported: {path}")
                continue
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown source type: {source_type}")

        if int(info["num_frames"]) < min_frames:
            warnings.append(
                f"skipped {source_type} with {info['num_frames']} frames; needs at least {min_frames}: {path}"
            )
            continue
        sample_index = len(records)
        source_counts[source_type] += 1
        records.append(
            {
                "sample_id": f"real_{sample_index:06d}",
                "source_type": source_type,
                "path": _metadata_path_for(path, metadata_path),
                "task_text": task_text,
                "num_frames": int(info["num_frames"]),
                "fps": int(info.get("fps", 5)),
                "height": int(info["height"]),
                "width": int(info["width"]),
                "source": source,
                "split": split,
            }
        )

    if not records:
        raise ValueError(
            f"No usable real-video samples found in {input_dir}; "
            f"supported stable inputs are .pt clips and frame folders with at least {min_frames} frames"
        )

    write_metadata_jsonl(metadata_path, records)
    return IndexBuildSummary(
        metadata_path=str(metadata_path),
        num_samples=len(records),
        source_counts=source_counts,
        warnings=warnings,
    ).to_dict()
