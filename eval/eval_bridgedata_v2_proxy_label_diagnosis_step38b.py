"""Evaluate and report Step38B proxy label learnability diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_proxy_label_diagnosis_step38b import STAGE, load_step38b_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_proxy_label_diagnosis_step38b.yaml"


def evaluate_step38b_proxy_label_diagnosis(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_step38b_config(config_path)
    summary = _read_json(config["output"]["diagnosis_summary_json"])
    decomposition = _read_json(config["output"]["label_decomposition_json"])
    cross_shard = _read_json(config["output"]["cross_shard_consistency_json"])
    gate = _read_json(config["output"]["gate_decision_json"])
    safety_gate = _safety_gate(summary, decomposition, gate)
    aggregate = summary.get("all_samples", {})
    result = {
        "stage": STAGE,
        "pass": bool(not summary.get("safe_stop", False) and summary.get("diagnosis_performed", False) and safety_gate),
        "safe_stop": bool(summary.get("safe_stop", False)),
        "reason": summary.get("reason"),
        "diagnosis_performed": bool(summary.get("diagnosis_performed", False)),
        "num_samples": int(summary.get("num_samples", 0)),
        "num_decomposition_rows": int(decomposition.get("num_rows", 0)),
        "temporal_component_dominates": bool(gate.get("temporal_component_dominates", False)),
        "spatial_residual_learnable_evidence": bool(gate.get("spatial_residual_learnable_evidence", False)),
        "spatial_residual_cross_shard_consistent": bool(gate.get("spatial_residual_cross_shard_consistent", False)),
        "spatial_residual_energy_ratio_mean": aggregate.get("residual_energy_ratio_mean"),
        "temporal_broadcast_r2_like_mean": aggregate.get("temporal_broadcast_r2_like_mean"),
        "spatial_entropy_mean": aggregate.get("per_frame_spatial_entropy_mean_mean"),
        "cross_shard_residual_cosine": cross_shard.get("cross_shard_residual_cosine"),
        "recommended_step39": gate.get("recommended_step39", {}),
        "selector_training_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "label_patch_detail_learnable_allowed": False,
        "training_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "new_dataset_download_performed": False,
        "new_model_download_performed": False,
        "tensorflow_required": False,
        "tensorflow_datasets_required": False,
        "extra_cloud_resources_used": False,
        "safety_gate_pass": bool(safety_gate),
    }
    _write_json(config["output"]["eval_json"], result)
    Path(config["output"]["eval_md"]).write_text(_eval_markdown(result), encoding="utf-8")
    _write_docs(summary, decomposition, cross_shard, gate, result)
    return result


def _safety_gate(summary: dict[str, Any], decomposition: dict[str, Any], gate: dict[str, Any]) -> bool:
    flags_false = [
        summary.get("training_performed", False),
        summary.get("optimizer_step_performed", False),
        summary.get("checkpoint_saved", False),
        summary.get("state_dict_saved", False),
        summary.get("data_token_shards_written", False),
        summary.get("data_importance_shards_written", False),
        summary.get("new_dataset_download_performed", False),
        summary.get("new_model_download_performed", False),
        decomposition.get("training_performed", False),
        decomposition.get("optimizer_step_performed", False),
        decomposition.get("checkpoint_saved", False),
        decomposition.get("state_dict_saved", False),
        gate.get("selector_training_allowed", False),
        gate.get("final_selector_training_allowed", False),
        gate.get("current_importance_training_allowed", False),
        gate.get("context_utility_claim_allowed", False),
        gate.get("label_patch_detail_learnable_allowed", False),
        gate.get("optimizer_step_performed", False),
        gate.get("checkpoint_saved", False),
        gate.get("state_dict_saved", False),
        gate.get("data_token_shards_written", False),
        gate.get("data_importance_shards_written", False),
        gate.get("new_dataset_download_performed", False),
        gate.get("new_model_download_performed", False),
    ]
    return not any(bool(flag) for flag in flags_false)


def _write_docs(
    summary: dict[str, Any],
    decomposition: dict[str, Any],
    cross_shard: dict[str, Any],
    gate: dict[str, Any],
    result: dict[str, Any],
) -> None:
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    aggregate = summary.get("all_samples", {})
    recommended = gate.get("recommended_step39", {})
    temporal_answer = (
        "yes" if result.get("temporal_component_dominates") else "not established"
    )
    residual_answer = (
        "yes" if result.get("spatial_residual_learnable_evidence") else "no, not enough evidence"
    )
    patch_answer = (
        "yes, only as another bounded smoke"
        if result.get("spatial_residual_learnable_evidence")
        else "no, redesign or denoise the proxy label first"
    )
    (docs_dir / "STEP38B_PROXY_LABEL_LEARNABILITY_DIAGNOSIS.md").write_text(
        "\n".join(
            [
                "# Step38B Proxy Label Learnability Diagnosis",
                "",
                "Step38B may diagnose and prepare evidence for proxy-supervised selector work.",
                "Step38B must not claim context utility.",
                "Step38B must not train final selector, current importance, VideoMAE, V-JEPA, VLM, world model, downstream model, RL, or action-conditioned model.",
                "Step38B must not use downstream task improvement as evidence.",
                "Step38B must not download extra datasets or models.",
                "Step38B keeps VideoMAE frozen by not loading or updating it.",
                "Step38B uses strict shard-aware splits only for leakage checks and cross-shard diagnosis.",
                "",
                "Questions:",
                f"1. Does the temporal component explain most variance? `{temporal_answer}`.",
                f"2. Is the spatial residual stable, concentrated, and cross-shard consistent? `{residual_answer}`.",
                f"3. Is the current patch-level label suitable for continued patch selector training? `{patch_answer}`.",
                f"4. Recommended Step39: `{recommended.get('name')}`.",
                "5. This step performed no selector/current-importance/final-selector/world-model training and no new data/model downloads.",
                "",
                "Key metrics:",
                f"- safe_stop: `{str(result.get('safe_stop')).lower()}`",
                f"- num_samples: `{result.get('num_samples')}`",
                f"- residual_energy_ratio_mean: `{aggregate.get('residual_energy_ratio_mean')}`",
                f"- temporal_broadcast_r2_like_mean: `{aggregate.get('temporal_broadcast_r2_like_mean')}`",
                f"- spatial_entropy_mean: `{aggregate.get('per_frame_spatial_entropy_mean_mean')}`",
                f"- cross_shard_residual_cosine: `{cross_shard.get('cross_shard_residual_cosine')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (docs_dir / "BRIDGEDATA_V2_PROXY_LABEL_DECOMPOSITION_REPORT.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Proxy Label Decomposition Report",
                "",
                "The label decomposition is statistical only:",
                "- proxy label: `[16,392]`",
                "- temporal component: `[16]`",
                "- temporal broadcast: `[16,392]`",
                "- spatial residual: `[16,392]`",
                "",
                f"- num_rows: `{decomposition.get('num_rows')}`",
                f"- all_samples: `{json.dumps(_compact_aggregate(aggregate), sort_keys=True)}`",
                f"- cross_shard: `{json.dumps(_compact_cross(cross_shard), sort_keys=True)}`",
                "",
                "Action, language, future tokens, current importance, and downstream labels are not used as diagnosis inputs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (docs_dir / "BRIDGEDATA_V2_STEP38B_DECISION.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Step38B Decision",
                "",
                f"- temporal_component_dominates: `{str(gate.get('temporal_component_dominates')).lower()}`",
                f"- spatial_residual_learnable_evidence: `{str(gate.get('spatial_residual_learnable_evidence')).lower()}`",
                f"- spatial_residual_cross_shard_consistent: `{str(gate.get('spatial_residual_cross_shard_consistent')).lower()}`",
                "- label_patch_detail_learnable_allowed: `false`",
                "- selector_training_allowed: `false`",
                "- final_selector_training_allowed: `false`",
                "- current_importance_training_allowed: `false`",
                "- context_utility_claim_allowed: `false`",
                "",
                "Recommended Step39:",
                f"- name: `{recommended.get('name')}`",
                f"- scope: `{recommended.get('scope')}`",
                f"- reason: `{recommended.get('reason')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _compact_aggregate(payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "num_rows",
        "residual_energy_ratio_mean",
        "temporal_energy_ratio_mean",
        "temporal_broadcast_r2_like_mean",
        "per_frame_spatial_entropy_mean_mean",
        "top256_mass_ratio_mean",
        "top256_residual_mass_ratio_mean",
        "temporal_component_dominates",
        "spatial_residual_learnable_evidence",
    ]
    return {key: payload.get(key) for key in keys}


def _compact_cross(payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "computed",
        "shards",
        "num_samples_by_shard",
        "cross_shard_residual_cosine",
        "cross_shard_residual_l1",
        "cross_shard_residual_consistent",
    ]
    return {key: payload.get(key) for key in keys}


def _eval_markdown(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step38B Eval",
            "",
            f"- pass: `{str(result.get('pass')).lower()}`",
            f"- safe_stop: `{str(result.get('safe_stop')).lower()}`",
            f"- safety_gate_pass: `{str(result.get('safety_gate_pass')).lower()}`",
            f"- temporal_component_dominates: `{str(result.get('temporal_component_dominates')).lower()}`",
            f"- spatial_residual_learnable_evidence: `{str(result.get('spatial_residual_learnable_evidence')).lower()}`",
            f"- recommended_step39: `{(result.get('recommended_step39') or {}).get('name')}`",
            "- selector_training_allowed: `false`",
            "- final_selector_training_allowed: `false`",
            "- current_importance_training_allowed: `false`",
            "- context_utility_claim_allowed: `false`",
            "- optimizer_step_performed: `false`",
            "- checkpoint_saved: `false`",
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
    print(json.dumps(evaluate_step38b_proxy_label_diagnosis(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
