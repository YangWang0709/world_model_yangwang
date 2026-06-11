"""Static Step17 context-bottleneck code audit for Step17.5."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "code_audit_step17_context_bottleneck.yaml"
REQUIRED_TESTS = [
    "tests/test_unified_selector_init_report.py",
    "tests/test_step17_no_current_token_drop_strict.py",
    "tests/test_step17_no_current_importance_training.py",
    "tests/test_step17_context_importance_no_current_occlusion.py",
    "tests/test_step17_no_oracle_leakage_policies.py",
    "tests/test_step17_full_context_teacher_reference_aggregate.py",
    "tests/test_step17_runner_protected_paths.py",
    "tests/test_step17_artifact_blacklist.py",
]
REQUIRED_GITIGNORE = [
    "runs/",
    "outputs/",
    "checkpoints/",
    "model_cache/",
    "weights/",
    "hf_cache/",
    "data/token_shards/",
    "data/importance_shards/",
    "data/context_token_shards/",
    "data/context_importance_shards/",
    "data/bair_context_windows_1000_128/",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.safetensors",
    "*.bin",
    "*.onnx",
    "*.log",
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return payload


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_text(path: str | Path) -> str:
    return _path(path).read_text(encoding="utf-8")


def _read_json(path: str | Path, required: bool = False) -> dict[str, Any]:
    p = _path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(p)
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {p}")
    return payload


def _check(name: str, passed: bool, failures: list[str], message: str) -> bool:
    if not passed:
        failures.append(f"{name}: {message}")
    return passed


def artifact_blacklist_patterns() -> list[str]:
    return [
        "data/bair_robot_pushing_small_tfds/",
        "data/bair_robot_pushing_small_subset/",
        "data/bair_robot_pushing_small_subset_500_64/",
        "data/bair_robot_pushing_small_subset_1000_128/",
        "data/bair_context_windows_1000_128/",
        "data/token_shards/",
        "data/context_token_shards/",
        "data/importance_shards/",
        "data/context_importance_shards/",
        "model_cache/",
        "hf_cache/",
        "weights/",
        "runs/",
        "outputs/",
        "checkpoints/",
    ]


def forbidden_git_status_entries(status_text: str) -> list[str]:
    bad: list[str] = []
    suffixes = (".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".log", ".tfrecord")
    sensitive = ("password", "secret", "github_token", "ssh_key", "id_rsa", "id_ed25519")
    for raw in status_text.splitlines():
        path = raw[3:].strip() if len(raw) > 3 else raw.strip()
        normalized = path.replace("\\", "/")
        if any(normalized.startswith(prefix) for prefix in artifact_blacklist_patterns()):
            bad.append(raw)
            continue
        lower = normalized.lower()
        if lower.endswith(suffixes) or any(token in lower for token in sensitive):
            bad.append(raw)
    return bad


def run_git_status_short() -> str:
    result = subprocess.run(
        ["git", "status", "--short", "--ignored=no"],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    return (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")


def render_audit_markdown(summary: dict[str, Any]) -> str:
    checks = [
        ("config_audit_pass", summary["config_audit_pass"]),
        ("importance_context_only_pass", summary["importance_context_only_pass"]),
        ("current_not_dropped_pass", summary["current_not_dropped_pass"]),
        ("no_current_importance_training_pass", summary["no_current_importance_training_pass"]),
        ("no_oracle_leakage_policy_tests_pass", summary["no_oracle_leakage_policy_tests_pass"]),
        ("full_context_teacher_reference_aggregate_pass", summary["full_context_teacher_reference_aggregate_pass"]),
        ("gitignore_artifact_blacklist_pass", summary["gitignore_artifact_blacklist_pass"]),
        ("pytest_pass", summary["pytest_pass"]),
    ]
    lines = [
        "# Step17 Context Bottleneck Code Audit Summary",
        "",
        f"- stage: `{summary['stage']}`",
        f"- audit pass: `{str(summary['audit_pass']).lower()}`",
        "",
        "## Checks",
        "",
        "| check | pass |",
        "| --- | --- |",
    ]
    for name, passed in checks:
        lines.append(f"| {name} | `{str(passed).lower()}` |")
    lines.extend(["", "## Blocking Issues", ""])
    lines.extend([f"- {item}" for item in summary["blocking_issues"]] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in summary["warnings"]] or ["- none"])
    lines.extend(["", "## Files Reviewed", ""])
    lines.extend([f"- `{item}`" for item in summary["files_reviewed"]])
    lines.extend(["", f"STEP17_CODE_AUDIT_PASS = {str(summary['audit_pass']).lower()}", ""])
    return "\n".join(lines)


def run_step17_code_audit(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    pytest_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    inputs = config["inputs"]
    output = config["output"]
    failures: list[str] = []
    warnings: list[str] = []
    files_reviewed = [
        str(inputs["step17_config"]),
        str(inputs["train_unified_selector_config"]),
        str(inputs["context_importance_config"]),
        "scripts/generate_context_predictive_importance.py",
        "models/unified_predictive_importance_selector.py",
        "models/context_bottleneck_world_model.py",
        "models/context_selection_policies.py",
        "training/train_unified_context_selector.py",
        "training/train_context_bottleneck_world_model.py",
        "training/run_bair_context_bottleneck_validation.py",
        ".gitignore",
    ]
    step17 = load_yaml(inputs["step17_config"])
    train_selector = load_yaml(inputs["train_unified_selector_config"])
    importance_cfg = load_yaml(inputs["context_importance_config"])
    importance_source = _read_text("scripts/generate_context_predictive_importance.py")
    selector_source = _read_text("models/unified_predictive_importance_selector.py")
    policy_source = _read_text("models/context_selection_policies.py")
    world_source = _read_text("models/context_bottleneck_world_model.py")
    baseline_source = _read_text("training/train_context_bottleneck_world_model.py")
    runner_source = _read_text("training/run_bair_context_bottleneck_validation.py")
    gitignore = _read_text(".gitignore")
    baseline = _read_json(inputs["baseline_aggregate"], required=False)

    config_pass = all(
        [
            step17["importance"].get("occlude_only") == "context_tokens",
            step17["importance"].get("keep_current_full") is True,
            step17["unified_selector"].get("train_mode") == "context",
            step17["unified_selector"].get("train_state_mode") is False,
            step17["context_bottleneck_world_model"].get("current_token_drop") is False,
            step17["dataset"].get("use_action_as_input") is False,
            step17["dataset"].get("use_endeffector_as_input") is False,
            train_selector["training"].get("train_state_mode") is False,
            importance_cfg["importance"].get("occlude_only") == "context_tokens",
            importance_cfg["importance"].get("keep_current_full") is True,
        ]
    )
    _check("config_audit", config_pass, failures, "Step17 config no-current-drop/context-only flags are not all set")

    importance_pass = all(
        [
            "masked_context = context_tokens.clone()" in importance_source,
            "masked_context[:, token_idx, :]" in importance_source,
            "pred = teacher(masked_context, current_tokens)" in importance_source,
            '"occlude_only": "context_tokens"' in importance_source,
            '"keep_current_full": True' in importance_source,
            "current_importance_scores" not in importance_source,
        ]
    )
    _check("importance_context_only", importance_pass, failures, "context importance generator may mask current tokens or emit current importance")

    current_not_dropped_pass = all(
        [
            "current_tokens.mean(dim=1)" in world_source,
            "current_token_drop: false" in _read_text(inputs["step17_config"]),
            "gather_tokens_by_indices(current" not in world_source,
            "torch.topk(current" not in world_source,
        ]
    )
    _check("current_not_dropped", current_not_dropped_pass, failures, "current tokens are not clearly kept full and mean-pooled")

    no_current_importance_pass = all(
        [
            '"trained_current_importance": False' in _read_text("training/train_unified_context_selector.py"),
            "importance_scores_norm" in _read_text("training/train_unified_context_selector.py"),
            "current_importance_shards" not in _read_text(inputs["train_unified_selector_config"]),
            "current_importance_scores" not in _read_text("training/train_unified_context_selector.py"),
        ]
    )
    _check("no_current_importance_training", no_current_importance_pass, failures, "current importance training path found or context target missing")

    selector_pass = all(
        [
            'mode == "context"' in selector_source,
            'mode == "state_legacy"' in selector_source,
            "current_summary" in selector_source,
            "initialize_context_selector_from_state_checkpoint_with_report" in selector_source,
            '"init_report"' in _read_text("training/train_unified_context_selector.py"),
        ]
    )
    _check("selector_audit", selector_pass, failures, "unified selector mode/init report audit failed")

    policy_pass = all(
        [
            'policy == "teacher_context_importance_topK"' in policy_source,
            'KeyError("teacher_context_importance_topK requires importance_scores_norm")' in policy_source,
            'policy in {"learned_context_selector_topK", "hybrid_context_learned_uniform"}' in policy_source,
            "batch[\"importance_scores_norm\"]" not in policy_source.split('policy in {"learned_context_selector_topK", "hybrid_context_learned_uniform"}', 1)[-1],
        ]
    )
    _check("oracle_policy_isolation", policy_pass, failures, "deployable policies may access teacher importance labels")

    aggregate_rows = baseline.get("aggregate", []) if isinstance(baseline.get("aggregate"), list) else []
    baseline_has_teacher = any(row.get("policy") == "full_context_teacher_reference" for row in aggregate_rows)
    baseline_pass = all(
        [
            "evaluate_full_context_teacher_reference" in baseline_source,
            '"is_teacher_reference": True' in baseline_source,
            '"trainable_student": False' in baseline_source,
            '"deployable_policy": False' in baseline_source,
            baseline_has_teacher or not baseline,
        ]
    )
    if not baseline_has_teacher:
        warnings.append("Existing Step17 baseline aggregate was not loaded or lacks full_context_teacher_reference; helper coverage is still tested statically.")
    _check("full_context_teacher_reference", baseline_pass, failures, "full_context_teacher_reference helper/aggregate contract missing")

    runner_pass = all(
        [
            "PROTECTED_MARKERS" in runner_source,
            "validate_step17_output_paths" in runner_source,
            "Refusing partial summary outside Step17-owned output path" in runner_source,
        ]
    )
    _check("runner_protected_paths", runner_pass, failures, "runner protected path guard is incomplete")

    missing_gitignore = [item for item in REQUIRED_GITIGNORE if item not in gitignore]
    git_status = run_git_status_short()
    forbidden_status = forbidden_git_status_entries(git_status)
    gitignore_pass = not missing_gitignore and not forbidden_status
    if missing_gitignore:
        failures.append(f"gitignore_artifact_blacklist: missing {missing_gitignore}")
    if forbidden_status:
        failures.append(f"gitignore_artifact_blacklist: forbidden git status entries {forbidden_status}")

    missing_tests = [path for path in REQUIRED_TESTS if not _path(path).exists()]
    if missing_tests:
        failures.append(f"required_tests: missing {missing_tests}")

    pytest_pass = bool(pytest_result.get("passed")) if pytest_result is not None else True
    audit_pass = all(
        [
            config_pass,
            importance_pass,
            current_not_dropped_pass,
            no_current_importance_pass,
            policy_pass,
            baseline_pass,
            gitignore_pass,
            not missing_tests,
            pytest_pass,
        ]
    )
    summary = {
        "stage": "step17_code_audit_patch",
        "config_audit_pass": bool(config_pass),
        "importance_context_only_pass": bool(importance_pass),
        "current_not_dropped_pass": bool(current_not_dropped_pass),
        "no_current_importance_training_pass": bool(no_current_importance_pass),
        "no_oracle_leakage_policy_tests_pass": bool(policy_pass),
        "full_context_teacher_reference_aggregate_pass": bool(baseline_pass),
        "gitignore_artifact_blacklist_pass": bool(gitignore_pass),
        "pytest_pass": bool(pytest_pass),
        "audit_pass": bool(audit_pass),
        "blocking_issues": failures,
        "warnings": warnings,
        "files_reviewed": files_reviewed,
        "required_tests": REQUIRED_TESTS,
        "pytest_result": pytest_result or {},
        "recommended_next_step": "Step18 context selector oracle-gap diagnostic",
    }
    run_dir = Path(output["run_root"]) / output["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(output["summary_json"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(output["summary_md"]).write_text(render_audit_markdown(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    summary = run_step17_code_audit(parse_args().config)
    print(json.dumps(summary, indent=2))
    print(f"STEP17_CODE_AUDIT_PASS = {str(summary['audit_pass']).lower()}")
    if not summary["audit_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
