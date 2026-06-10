"""Small in-memory token bank placeholder for future task memory work."""

from __future__ import annotations

from typing import Any


class MemoryBank:
    """A lightweight memory container with the interface needed by later phases."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._entries.clear()

    def write(self, tokens, metadata: dict[str, Any] | None = None) -> None:
        self._entries.append({"tokens": tokens, "metadata": metadata or {}})

    def read(self, query=None, topk: int | None = None):
        del query
        if topk is None:
            return list(self._entries)
        return list(self._entries[-topk:])

