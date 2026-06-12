from pathlib import Path

import yaml

from data.bridgedata_v2_tfds_proxy_temporal_selector_step34 import (
    build_step34_selector_plan,
    env_isaaclab_tensorflow_guard,
)


def test_step34_plan_from_fake_step33b_outputs_keeps_training_disallowed(tmp_path: Path):
    config = _fake_config(tmp_path)
    plan = build_step34_selector_plan(config, run_dry_run=False)

    assert plan["safe_stop"] is False
    assert plan["decision"]["may_prepare_proxy_supervised_selector_training"] is True
    assert plan["decision"]["selector_training_allowed"] is False
    assert plan["decision"]["current_importance_training_allowed"] is False
    assert plan["decision"]["context_utility_claim_allowed"] is False
    assert plan["decision"]["future_selector_training_gate_ready"] is False
    assert plan["step33b_evidence"]["proxy_signal_stable_across_shards"] is True
    assert plan["step33b_evidence"]["dataset_bias_detected"] is False
    assert plan["leakage_checks"]["no_language_or_trajectory_leakage"] is True
    assert plan["future_training_gates"]["selector_beats_random_baseline"]["status"] == "pending"
    assert plan["future_training_gates"]["selector_beats_current_only_baseline"]["status"] == "pending"


def test_step34_env_guard_declares_tensorflow_not_required():
    guard = env_isaaclab_tensorflow_guard()
    assert guard["tensorflow_required"] is False
    assert guard["tensorflow_datasets_required"] is False


def _fake_config(tmp_path: Path) -> dict:
    base = yaml.safe_load(
        Path("configs/bridgedata_v2_tfds_proxy_temporal_selector_step34.yaml").read_text(encoding="utf-8")
    )
    run = tmp_path / "step33b"
    run.mkdir()
    files = {
        "step33b_eval_json": {"pass": True, "safe_stop": False, "safety_gate_pass": True},
        "step33b_decision_json": {
            "proxy_signal_stable_across_shards": True,
            "context_utility_claim_allowed": False,
            "selector_training_allowed": False,
            "current_importance_training_allowed": False,
        },
        "step33b_dataset_bias_json": {
            "dataset_bias_detected": False,
            "distribution_shift_flags": {
                "action_distribution_shift": False,
                "episode_length_shift": False,
                "language_hash_shift": False,
                "proxy_temporal_concentration_shift": False,
                "proxy_topk_mass_shift": False,
            },
            "language_hash_comparable": False,
            "raw_language_text_saved": False,
        },
        "step33b_within_shard_results_json": _metric_summary(0.157),
        "step33b_cross_shard_results_json": _metric_summary(0.385, rows=6),
        "step33b_mixed_shard_results_json": _metric_summary(0.369),
        "step33b_shard_split_json": {
            "split_seeds": [42],
            "splits": [
                {
                    "split_seed": 42,
                    "train_sample_ids": ["a", "b"],
                    "val_sample_ids": ["c"],
                    "train_trajectories": ["ta", "tb"],
                    "val_trajectories": ["tc"],
                    "train_val_trajectory_disjoint": True,
                }
            ],
        },
        "step33b_token_summary_json": {"token_shapes_ok": True},
        "step33b_importance_summary_json": {"topk_mass_mean": 0.205592},
    }
    for key, payload in files.items():
        path = run / f"{key}.json"
        path.write_text(__import__("json").dumps(payload), encoding="utf-8")
        base["input"][key] = str(path)
    for key in ["step33b_token_manifest_jsonl", "step33b_importance_manifest_jsonl"]:
        path = run / f"{key}.jsonl"
        path.write_text("", encoding="utf-8")
        base["input"][key] = str(path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    base["input"]["local_videomae_model"] = str(model_dir)
    base["output"]["run_dir"] = str(tmp_path / "step34")
    base["output"]["plan_summary_json"] = str(tmp_path / "step34" / "plan.json")
    base["output"]["dry_run_summary_json"] = str(tmp_path / "step34" / "dry.json")
    base["output"]["eval_json"] = str(tmp_path / "step34" / "eval.json")
    base["output"]["eval_md"] = str(tmp_path / "step34" / "eval.md")
    base["output"]["plan_doc_md"] = str(tmp_path / "docs" / "plan.md")
    base["output"]["gates_doc_md"] = str(tmp_path / "docs" / "gates.md")
    return base


def _metric_summary(gain: float, rows: int = 3) -> dict:
    return {
        "num_rows": rows,
        "proxy_beats_current_fraction": 1.0,
        "proxy_beats_random_fraction": 1.0,
        "proxy_gain_over_current_mean": gain,
        "proxy_gain_over_random_mean": gain / 2.0,
        "full_context_noise_confirmed": True,
    }
