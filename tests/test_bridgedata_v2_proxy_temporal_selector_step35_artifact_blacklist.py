import subprocess
from pathlib import Path


FORBIDDEN_STAGED_PARTS = (
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
FORBIDDEN_SUFFIXES = (
    ".tfrecord",
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".jpg",
    ".jpeg",
    ".png",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".log",
)


def test_gitignore_contains_step35_required_artifact_patterns():
    text = Path(".gitignore").read_text(encoding="utf-8")
    for pattern in [
        "runs/",
        "outputs/",
        "checkpoints/",
        "weights/",
        "model_cache/",
        "hf_cache/",
        "data/token_shards/",
        "data/importance_shards/",
        "data/bridgedata_v2_tfds_mini/",
        "data/bridgedata_v2_tfds_cache/",
        "data/bridgedata_v2_tfds_shards/",
        "data/bridgedata_v2_tfds_downloads/",
        "*.tfrecord",
        "*.tfrecord-*",
        "*.npz",
        "*.npy",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "*.safetensors",
        "*.zip",
        "*.tar",
        "*.tar.gz",
        "*.tgz",
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.mp4",
        "*.avi",
        "*.mov",
        "*.mkv",
        "*.log",
    ]:
        assert pattern in text


def test_step35_staged_files_do_not_include_large_or_secret_artifacts():
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return
    staged = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    bad = []
    for path in staged:
        lowered = path.lower()
        if any(part in lowered for part in FORBIDDEN_STAGED_PARTS):
            bad.append(path)
        if lowered.endswith(FORBIDDEN_SUFFIXES):
            bad.append(path)
        if any(secret_word in lowered for secret_word in ("password", "secret", "github_token", "access_token")):
            bad.append(path)
    assert bad == []
