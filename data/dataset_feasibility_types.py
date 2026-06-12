"""Typed containers for Step19 long-context dataset feasibility summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


BoolLike = bool | None


@dataclass(frozen=True)
class DatasetFeasibilityEntry:
    dataset: str
    priority: str
    full_download_allowed: bool
    local_cache_exists: bool
    supports_long_context_likely: BoolLike
    supports_goal_image_likely: BoolLike
    supports_language_likely: BoolLike
    supports_action_likely: BoolLike
    supports_multicam_likely: BoolLike
    estimated_local_difficulty: str
    recommended_role: str
    recommended_next_action: str
    evidence: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalResourceSnapshot:
    disk_free_gb: float
    disk_total_gb: float
    ram_total_gb: float | None
    ram_available_gb: float | None
    gpu_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
