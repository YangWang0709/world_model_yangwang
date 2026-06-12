import subprocess
from pathlib import Path


def test_gitignore_still_contains_tfds_artifact_blacklist():
    text = Path(".gitignore").read_text(encoding="utf-8")
    required = [
        "data/bridgedata_v2_tfds_mini/",
        "data/bridgedata_v2_tfds_cache/",
        "data/bridgedata_v2_tfds_shards/",
        "data/bridgedata_v2_tfds_downloads/",
        "datasets/bridgedata_v2_tfds_mini/",
        "*.tfrecord",
        "*.tfrecord-*",
        "runs/",
        "*.pt",
    ]
    missing = [item for item in required if item not in text]
    assert not missing


def test_staged_files_exclude_tfds_shards_runs_and_weights():
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], check=False, capture_output=True, text=True)
    staged = result.stdout.splitlines() if result.returncode == 0 else []
    forbidden_prefixes = (
        "runs/",
        "outputs/",
        "checkpoints/",
        "data/bridgedata_v2_tfds_mini/",
        "data/bridgedata_v2_tfds_cache/",
        "data/bridgedata_v2_tfds_shards/",
        "data/bridgedata_v2_tfds_downloads/",
        "datasets/bridgedata_v2_tfds_mini/",
        "data/token_shards/",
        "data/importance_shards/",
    )
    forbidden_suffixes = (
        ".tfrecord",
        ".zip",
        ".tar",
        ".tar.gz",
        ".tgz",
        ".jpg",
        ".jpeg",
        ".png",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".npy",
        ".npz",
        ".hdf5",
        ".h5",
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".bin",
        ".onnx",
        ".log",
    )
    bad = [
        path
        for path in staged
        if path.startswith(forbidden_prefixes)
        or path.lower().endswith(forbidden_suffixes)
        or ".tfrecord-" in path.lower()
    ]
    assert not bad
