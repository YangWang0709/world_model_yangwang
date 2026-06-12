"""Create text-only templates for Step22 user-provided BridgeData V2 subsets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_user_subset_ingestion_step22.yaml"


def create_user_subset_template(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_record = {
        "trajectory_id": "traj_000001",
        "split": "train",
        "num_frames": 40,
        "image_dir": "traj_000001/images",
        "camera_names": ["main"],
        "actions_path": "traj_000001/actions.npy",
        "language_instruction": "put the object in the bowl",
        "goal_image_path": "traj_000001/goal.jpg",
        "task_id": "put_object_in_bowl",
        "environment_id": "kitchen_01",
        "metadata": {"source": "user_provided_tiny_subset"},
    }
    metadata = {
        "trajectory_id": "traj_000001",
        "split": "train",
        "language_instruction": "put the object in the bowl",
        "task_id": "put_object_in_bowl",
        "environment_id": "kitchen_01",
        "camera_names": ["main"],
        "actions_path": "traj_000001/actions.npy",
        "goal_image_path": "traj_000001/goal.jpg",
        "metadata": {"source": "user_provided_tiny_subset"},
    }
    files = {
        "manifest_template.jsonl": json.dumps(manifest_record, sort_keys=True) + "\n",
        "metadata_template.json": json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        "README_USER_SUBSET_FORMAT.md": _template_readme(),
        "example_tree.txt": _example_tree(),
    }
    written: list[str] = []
    for name, content in files.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        written.append(str(path))
    return {
        "stage": "bridgedata_v2_user_subset_ingestion_step22",
        "template_dir": str(output_dir),
        "files_written": written,
        "text_only": True,
        "image_files_created": [],
        "pt_files_created": [],
        "npy_files_created": [],
        "download_performed": False,
    }


def run_from_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return create_user_subset_template(Path(config["template"]["output_dir"]))


def _template_readme() -> str:
    return "\n".join(
        [
            "# BridgeData V2 User Subset Format",
            "",
            "Place your tiny user-provided subset under:",
            "",
            "```text",
            "data/bridgedata_v2_tiny_user_subset/",
            "  manifest.jsonl",
            "  traj_000001/",
            "    images/",
            "      frame_000000.jpg",
            "      frame_000001.jpg",
            "      ...",
            "    metadata.json",
            "    actions.npy or actions.json optional",
            "    goal.jpg optional",
            "```",
            "",
            "Rules:",
            "",
            "- at least 1 trajectory",
            "- each trajectory needs at least 24 frames",
            "- action, language, and goal image are metadata only",
            "- do not commit this directory to git",
            "- this template intentionally creates no images, no `.npy`, and no `.pt` files",
            "",
        ]
    )


def _example_tree() -> str:
    return "\n".join(
        [
            "data/bridgedata_v2_tiny_user_subset/",
            "  manifest.jsonl",
            "  traj_000001/",
            "    images/",
            "      frame_000000.jpg",
            "      frame_000001.jpg",
            "      ...",
            "    metadata.json",
            "    actions.npy",
            "    goal.jpg",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        summary = run_from_config(args.config)
    else:
        summary = create_user_subset_template(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
