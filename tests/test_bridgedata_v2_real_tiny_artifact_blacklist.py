import subprocess
from pathlib import Path


def test_gitignore_contains_real_tiny_data_and_archive_blacklist():
    text = Path(".gitignore").read_text(encoding="utf-8")
    required = [
        "data/bridgedata_v2_real_tiny_downloads/",
        "data/bridgedata_v2_real_tiny_subset/",
        "data/bridgedata_v2_real_tiny_extracted/",
        "data/bridgedata_v2_tiny_user_subset/",
        "data/bridgedata_v2_tiny_subset/",
        "datasets/bridgedata_v2_real_tiny/",
        "*.zip",
        "*.tar",
        "*.tar.gz",
        "*.tgz",
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.npy",
        "*.npz",
        "*.hdf5",
        "*.h5",
        "*.tfrecord",
    ]
    missing = [item for item in required if item not in text]
    assert not missing


def test_staged_files_exclude_real_tiny_artifacts_and_runs():
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], check=False, capture_output=True, text=True)
    staged = result.stdout.splitlines() if result.returncode == 0 else []
    forbidden_prefixes = (
        "runs/",
        "data/bridgedata_v2_real_tiny_downloads/",
        "data/bridgedata_v2_real_tiny_extracted/",
        "data/bridgedata_v2_real_tiny_subset/",
        "data/bridgedata_v2/",
        "data/bridge_data_v2/",
        "datasets/bridgedata_v2/",
        "data/token_shards/",
        "data/importance_shards/",
    )
    forbidden_suffixes = (
        ".zip",
        ".tar",
        ".tar.gz",
        ".tgz",
        ".jpg",
        ".jpeg",
        ".png",
        ".npy",
        ".npz",
        ".hdf5",
        ".h5",
        ".tfrecord",
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
        if path.startswith(forbidden_prefixes) or path.lower().endswith(forbidden_suffixes)
    ]
    assert not bad
