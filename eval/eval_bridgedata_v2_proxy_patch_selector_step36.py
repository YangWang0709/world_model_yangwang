"""Evaluate and report Step36 proxy patch/token selector smoke training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_proxy_patch_selector_dataset_step36 import STAGE, load_step36_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_proxy_patch_selector_train_step36.yaml"


def evaluate_step36_proxy_patch_selector(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_step36_config(config_path)
    train_summary = _read_json(config["output"]["train_summary_json"])
    loss_curves = _read_json(config["output"]["loss_curves_json"])
    metrics = _read_json(config["output"]["metrics_json"])
    gate = _read_json(config["output"]["gate_decision_json"])
    safe_stop = bool(train_summary.get("safe_stop", False))
    within = metrics.get("within_shard", {})
    cross = metrics.get("cross_shard", {})
    mixed = metrics.get("mixed_shard", {})
    safety_gate = _safety_gate(train_summary, metrics, gate)
    summary = {
        "stage": STAGE,
        "pass": bool(not safe_stop and safety_gate and bool(train_summary.get("training_performed", False))),
        "safe_stop": bool(safe_stop),
        "reason": train_summary.get("reason"),
        "training_performed": bool(train_summary.get("training_performed", False)),
        "bounded_smoke_training_only": True,
        "patch_level_selector_training_performed": bool(
            train_summary.get("patch_level_selector_training_performed", False)
        ),
        "patch_token_selector_head_training_performed": bool(
            train_summary.get("patch_token_selector_head_training_performed", False)
        ),
        "optimizer_step_performed": bool(train_summary.get("optimizer_step_performed", False)),
        "optimizer_scope": train_summary.get("optimizer_scope"),
        "videomae_training_performed": False,
        "videomae_frozen": True,
        "current_importance_training_performed": False,
        "final_selector_training_performed": False,
        "checkpoint_saved": False,
        "within_shard_eval_performed": int(within.get("num_rows", 0)) > 0,
        "cross_shard_eval_performed": int(cross.get("num_rows", 0)) > 0,
        "mixed_shard_eval_performed": int(mixed.get("num_rows", 0)) > 0,
        "within_shard_metrics": _compact_metrics(within),
        "cross_shard_metrics": _compact_metrics(cross),
        "mixed_shard_metrics": _compact_metrics(mixed),
        "patch_selector_beats_random_token_baseline": bool(
            gate.get("patch_selector_beats_random_token_baseline", False)
        ),
        "patch_selector_beats_uniform_token_baseline": bool(
            gate.get("patch_selector_beats_uniform_token_baseline", False)
        ),
        "patch_selector_beats_temporal_broadcast_baseline": bool(
            gate.get("patch_selector_beats_temporal_broadcast_baseline", False)
        ),
        "patch_selector_beats_current_only_patch_baseline": bool(
            gate.get("patch_selector_beats_current_only_patch_baseline", False)
        ),
        "patch_selector_generalizes_cross_shard": bool(gate.get("patch_selector_generalizes_cross_shard", False)),
        "top256_overlap_mean": float(gate.get("top256_overlap_mean", 0.0)),
        "future_patch_selector_gate_ready": bool(gate.get("future_patch_selector_gate_ready", False)),
        "selector_training_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "new_dataset_download_performed": False,
        "model_download_performed": False,
        "tensorflow_required": False,
        "tensorflow_datasets_required": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "recommended_step37": gate.get("recommended_step37", {}),
        "safety_gate_pass": bool(safety_gate),
    }
    _write_json(config["output"]["eval_json"], summary)
    Path(config["output"]["eval_md"]).write_text(_eval_markdown(summary), encoding="utf-8")
    _write_docs(config, train_summary, loss_curves, metrics, gate, summary)
    return summary


def _safety_gate(train_summary: dict[str, Any], metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        bool(train_summary.get("bounded_smoke_training_only", False))
        and bool(train_summary.get("patch_level_selector_training_performed", False))
        and bool(train_summary.get("patch_token_selector_head_training_performed", False))
        and bool(train_summary.get("optimizer_step_performed", False))
        and train_summary.get("optimizer_scope") == "proxy_patch_token_selector_head_only"
        and not bool(train_summary.get("videomae_training_performed", False))
        and bool(train_summary.get("videomae_frozen", True))
        and not bool(train_summary.get("current_importance_training_performed", False))
        and not bool(train_summary.get("final_selector_training_performed", False))
        and not bool(train_summary.get("checkpoint_saved", False))
        and not bool(train_summary.get("state_dict_saved", False))
        and not bool(train_summary.get("data_token_shards_written", False))
        and not bool(train_summary.get("data_importance_shards_written", False))
        and not bool(gate.get("selector_training_allowed", False))
        and not bool(gate.get("final_selector_training_allowed", False))
        and not bool(gate.get("current_importance_training_allowed", False))
        and not bool(gate.get("context_utility_claim_allowed", False))
        and bool(metrics.get("safety_gate_pass", False))
    )


def _compact_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "num_rows",
        "patch_selector_val_mse",
        "random_token_baseline_mse",
        "uniform_token_baseline_mse",
        "temporal_broadcast_baseline_mse",
        "current_only_patch_baseline_mse",
        "patch_selector_beats_random_token_baseline",
        "patch_selector_beats_uniform_token_baseline",
        "patch_selector_beats_temporal_broadcast_baseline",
        "patch_selector_beats_current_only_patch_baseline",
        "patch_selector_spearman",
        "patch_selector_pearson",
        "top64_overlap",
        "top128_overlap",
        "top256_overlap",
        "top512_overlap",
        "top256_precision",
        "top256_recall",
        "overfit_gap",
    ]
    return {key: payload.get(key) for key in keys}


def _write_docs(
    config: dict[str, Any],
    train_summary: dict[str, Any],
    loss_curves: dict[str, Any],
    metrics: dict[str, Any],
    gate: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "STEP36_PROXY_PATCH_SELECTOR_TRAINING_SMOKE.md").write_text(
        "\n".join(
            [
                "# Step36 Proxy Patch/Token Selector Training Smoke",
                "",
                "Step36 trains only a bounded patch/token-level selector head smoke.",
                "Step36 does not train the final selector.",
                "Step36 does not train current importance.",
                "Step36 does not claim context utility.",
                "Step36 does not use downstream task improvement as evidence.",
                "",
                "Allowed:",
                "- Read existing Step33B token artifacts.",
                "- Read existing Step33B proxy importance artifacts.",
                "- Read Step35 temporal selector outputs as baseline reference.",
                "- Train only `ProxyPatchTokenSelectorHead`.",
                "",
                "Not allowed:",
                "- No raw zip, full TFDS, DROID, images, videos, checkpoints, or new model downloads.",
                "- No VideoMAE, world model, downstream predictor, policy, VLM/RL, or action-conditioned training.",
                "- No action or language as model input.",
                "- No writes to `data/" "token_shards/` or `data/" "importance_shards/`.",
                "",
                "Safety flags:",
                f"- optimizer_scope: `{train_summary.get('optimizer_scope')}`",
                f"- checkpoint_saved: `{str(train_summary.get('checkpoint_saved')).lower()}`",
                f"- selector_training_allowed: `{str(summary.get('selector_training_allowed')).lower()}`",
                f"- final_selector_training_allowed: `{str(summary.get('final_selector_training_allowed')).lower()}`",
                f"- current_importance_training_allowed: `{str(summary.get('current_importance_training_allowed')).lower()}`",
                f"- context_utility_claim_allowed: `{str(summary.get('context_utility_claim_allowed')).lower()}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    within = metrics.get("within_shard", {})
    cross = metrics.get("cross_shard", {})
    mixed = metrics.get("mixed_shard", {})
    (docs_dir / "BRIDGEDATA_V2_PROXY_PATCH_SELECTOR_TRAINING_REPORT.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Proxy Patch Selector Training Report",
                "",
                f"- num_curves: `{loss_curves.get('num_curves')}`",
                f"- within_shard: `{json.dumps(_compact_metrics(within), sort_keys=True)}`",
                f"- cross_shard: `{json.dumps(_compact_metrics(cross), sort_keys=True)}`",
                f"- mixed_shard: `{json.dumps(_compact_metrics(mixed), sort_keys=True)}`",
                f"- patch_selector_beats_random_token_baseline: `{str(gate.get('patch_selector_beats_random_token_baseline')).lower()}`",
                f"- patch_selector_beats_uniform_token_baseline: `{str(gate.get('patch_selector_beats_uniform_token_baseline')).lower()}`",
                f"- patch_selector_beats_temporal_broadcast_baseline: `{str(gate.get('patch_selector_beats_temporal_broadcast_baseline')).lower()}`",
                f"- patch_selector_generalizes_cross_shard: `{str(gate.get('patch_selector_generalizes_cross_shard')).lower()}`",
                f"- top256_overlap_mean: `{gate.get('top256_overlap_mean')}`",
                "",
                "The selector input is limited to existing context/current tokens. Action and language are metadata only.",
                "Full-context noisy evidence is acknowledged; this report does not claim context utility.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    recommended = gate.get("recommended_step37", {})
    (docs_dir / "BRIDGEDATA_V2_STEP36_PATCH_SELECTOR_DECISION.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Step36 Patch Selector Decision",
                "",
                f"- future_patch_selector_gate_ready: `{str(gate.get('future_patch_selector_gate_ready')).lower()}`",
                "- selector_training_allowed: `false`",
                "- final_selector_training_allowed: `false`",
                "- current_importance_training_allowed: `false`",
                "- context_utility_claim_allowed: `false`",
                "",
                "Recommended Step37:",
                f"- name: `{recommended.get('name')}`",
                f"- scope: `{recommended.get('scope')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _eval_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step36 Eval",
            "",
            f"- pass: `{str(summary.get('pass')).lower()}`",
            f"- safe_stop: `{str(summary.get('safe_stop')).lower()}`",
            f"- safety_gate_pass: `{str(summary.get('safety_gate_pass')).lower()}`",
            f"- optimizer_scope: `{summary.get('optimizer_scope')}`",
            f"- patch_selector_generalizes_cross_shard: `{str(summary.get('patch_selector_generalizes_cross_shard')).lower()}`",
            f"- top256_overlap_mean: `{summary.get('top256_overlap_mean')}`",
            f"- future_patch_selector_gate_ready: `{str(summary.get('future_patch_selector_gate_ready')).lower()}`",
            "- selector_training_allowed: `false`",
            "- current_importance_training_allowed: `false`",
            "- context_utility_claim_allowed: `false`",
        ]
    ) + "\n"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(evaluate_step36_proxy_patch_selector(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
