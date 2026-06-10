"""Check the runtime environment and write docs/ENV_REPORT_REMOTE.md."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "docs" / "ENV_REPORT_REMOTE.md"


def optional_import(name: str):
    try:
        module = importlib.import_module(name)
        return True, module, None
    except Exception as exc:  # pragma: no cover - report path
        return False, None, f"{type(exc).__name__}: {exc}"


def build_report() -> str:
    torch_ok, torch_mod, torch_error = optional_import("torch")
    numpy_ok, numpy_mod, numpy_error = optional_import("numpy")
    yaml_ok, yaml_mod, yaml_error = optional_import("yaml")

    cuda_available = False
    cuda_device_count = 0
    gpu_name = "none"
    torch_version = "not importable"
    if torch_ok:
        torch_version = getattr(torch_mod, "__version__", "unknown")
        cuda_available = bool(torch_mod.cuda.is_available())
        cuda_device_count = int(torch_mod.cuda.device_count())
        if cuda_available and cuda_device_count > 0:
            gpu_name = torch_mod.cuda.get_device_name(0)

    lines = [
        "# Remote Environment Report",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Python executable | `{sys.executable}` |",
        f"| Python version | `{sys.version.split()[0]}` |",
        f"| Platform | `{platform.platform()}` |",
        f"| Current working directory | `{Path.cwd()}` |",
        f"| Project root | `{PROJECT_ROOT}` |",
        f"| Conda env | `{os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}` |",
        f"| torch import | `{torch_ok}` |",
        f"| torch version | `{torch_version}` |",
        f"| CUDA available | `{cuda_available}` |",
        f"| torch.cuda.device_count | `{cuda_device_count}` |",
        f"| GPU name | `{gpu_name}` |",
        f"| numpy import | `{numpy_ok}` |",
        f"| numpy version | `{getattr(numpy_mod, '__version__', 'not importable') if numpy_ok else 'not importable'}` |",
        f"| yaml import | `{yaml_ok}` |",
        f"| yaml version | `{getattr(yaml_mod, '__version__', 'not importable') if yaml_ok else 'not importable'}` |",
    ]

    if torch_error or numpy_error or yaml_error:
        lines.extend(["", "## Import Errors", ""])
        if torch_error:
            lines.append(f"- torch: `{torch_error}`")
        if numpy_error:
            lines.append(f"- numpy: `{numpy_error}`")
        if yaml_error:
            lines.append(f"- yaml: `{yaml_error}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"ENV_REPORT_WRITTEN = {REPORT_PATH}")


if __name__ == "__main__":
    main()

