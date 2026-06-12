import subprocess
from pathlib import Path

import yaml


FORBIDDEN_PARTS = (
    "data/bridgedata_v2_tfds_mini/",
    "data/bridgedata_v2_tfds_cache/",
    "data/bridgedata_v2_tfds_shards/",
    "data/bridgedata_v2_tfds_downloads/",
    "data/token_shards/",
    "data/importance_shards/",
    "datasets/",
    "runs/",
    "outputs/",
    "checkpoints/",
    "model_cache/",
    "weights/",
    "hf_cache/",
)
FORBIDDEN_SUFFIXES = (".tfrecord", ".npz", ".npy", ".pt", ".pth", ".ckpt", ".safetensors", ".zip", ".tar", ".gz", ".tgz", ".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv", ".log")


def test_step31_gitignore_contains_artifact_patterns():
    text = Path(".gitignore").read_text(encoding="utf-8")
    for pattern in ["runs/", "outputs/", "checkpoints/", "weights/", "hf_cache/", "data/token_shards/", "data/importance_shards/", "*.tfrecord", "*.pt", "*.npz", "*.npy", "*.log"]:
        assert pattern in text


def test_step31_outputs_are_ignored_run_dir_not_data_or_checkpoints():
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_teacher_temporal_horizon_diagnosis_step31.yaml").read_text(encoding="utf-8"))
    output_text = "\n".join(config["output"].values())
    assert "runs/bridgedata_v2_tfds_teacher_temporal_horizon_diagnosis_step31_v1" in output_text
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text
    assert "checkpoints" not in output_text


def test_step31_staged_files_do_not_include_large_or_secret_artifacts():
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return
    staged = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    bad = []
    for path in staged:
        lowered = path.lower()
        if any(part in lowered for part in FORBIDDEN_PARTS):
            bad.append(path)
        if lowered.endswith(FORBIDDEN_SUFFIXES):
            bad.append(path)
        if any(word in lowered for word in ("password", "secret", "github_token", "access_token")):
            bad.append(path)
    assert bad == []
