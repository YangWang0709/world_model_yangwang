"""V-JEPA wrapper placeholder for Step 9B."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class VJEPAWrapper:
    """Import-safe interface placeholder for future local V-JEPA integration."""

    encoder_name = "vjepa"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        repo_path = self.config.get("repo_path")
        checkpoint_path = self.config.get("checkpoint_path")
        model_config_path = self.config.get("model_config_path")
        self.availability = {
            "available": False,
            "reason": "V-JEPA is a Step 9B placeholder; no weights are downloaded or loaded",
            "repo_path_exists": bool(repo_path and Path(str(repo_path)).exists()),
            "checkpoint_path_exists": bool(checkpoint_path and Path(str(checkpoint_path)).exists()),
            "model_config_path_exists": bool(model_config_path and Path(str(model_config_path)).exists()),
        }

    def is_available(self) -> bool:
        return False

    def encode(self, *args, **kwargs):
        raise NotImplementedError(
            "V-JEPA wrapper is an import-safe placeholder in Step 9B. "
            "Provide local code and weights in a later step before calling encode()."
        )
