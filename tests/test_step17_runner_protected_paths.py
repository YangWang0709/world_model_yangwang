from copy import deepcopy
from pathlib import Path

import pytest

from training.run_bair_context_bottleneck_validation import (
    DEFAULT_CONFIG,
    is_step17_owned_output_path,
    load_yaml,
    validate_step17_output_paths,
    write_partial_summary,
)


def test_step17_validate_output_paths_accepts_owned_paths():
    config = load_yaml(DEFAULT_CONFIG)
    validate_step17_output_paths(config)
    assert is_step17_owned_output_path(config["tokens"]["output_root"])
    assert is_step17_owned_output_path(config["importance"]["output_root"])


def test_step17_validate_output_paths_rejects_prior_step_dirs():
    config = load_yaml(DEFAULT_CONFIG)
    for bad in [
        "/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_1000_128_weighted_mse_multiseed_v1",
        "/home/ubuntu22/tgpawb_world_model/runs/downstream_utilization_bair_1000_128_v1",
        "/home/ubuntu22/tgpawb_world_model/data/token_shards/bair_videomae_1000_128",
    ]:
        mutated = deepcopy(config)
        mutated["output"]["run_name"] = Path(bad).name
        mutated["output"]["run_root"] = str(Path(bad).parent)
        with pytest.raises(ValueError):
            validate_step17_output_paths(mutated)


def test_step17_partial_summary_rejects_protected_project_dirs():
    bad = "/home/ubuntu22/tgpawb_world_model/runs/downstream_utilization_bair_1000_128_v1"
    with pytest.raises(ValueError):
        write_partial_summary(bad, current_step="x", completed_steps=[], status="failed")

