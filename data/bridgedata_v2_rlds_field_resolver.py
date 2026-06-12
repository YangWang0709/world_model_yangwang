"""Resolve BridgeData V2 RLDS fields for later TFDS token extraction."""

from __future__ import annotations

from typing import Any


DEFAULT_POLICY = {
    "image": {
        "required_prefix": "steps/observation/",
        "include_patterns": ["steps/observation/image_"],
        "exclude_patterns": ["episode_metadata/has_image_", "episode_metadata/"],
        "preferred_order": [
            "steps/observation/image_0",
            "steps/observation/image_1",
            "steps/observation/image_2",
            "steps/observation/image_3",
        ],
    },
    "action": {"preferred_order": ["steps/action"], "use_as_input": False},
    "language": {
        "preferred_order": [
            "steps/language_instruction",
            "steps/natural_language_instruction",
            "steps/language_embedding",
        ],
        "prefer_text_over_embedding": True,
        "use_as_input": False,
    },
    "goal": {
        "preferred_order": ["steps/observation/goal_image", "steps/goal_image", "steps/goal"],
        "use_as_input": False,
    },
}


def resolve_rlds_fields(candidate_fields: dict[str, list[str]], policy: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = _merge_policy(policy)
    image = resolve_image_field(candidate_fields.get("image_fields", []), merged["image"])
    action = resolve_action_field(candidate_fields.get("action_fields", []), merged["action"])
    language = resolve_language_field(candidate_fields.get("language_fields", []), merged["language"])
    goal = resolve_goal_field(candidate_fields.get("goal_fields", []), merged["goal"])
    blocking_errors = []
    warnings = []
    if not image["image_field_valid"]:
        blocking_errors.extend(image["blocking_errors"])
    warnings.extend(image["warnings"])
    warnings.extend(action["warnings"])
    warnings.extend(language["warnings"])
    warnings.extend(goal["warnings"])
    return {
        **image,
        **action,
        **language,
        **goal,
        "blocking_errors": blocking_errors,
        "warnings": warnings,
    }


def resolve_image_field(image_fields: list[str], policy: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**DEFAULT_POLICY["image"], **(policy or {})}
    candidates = sorted({str(field) for field in image_fields})
    rejected = []
    valid = []
    for field in candidates:
        reason = _image_rejection_reason(field, cfg)
        if reason is None:
            valid.append(field)
        else:
            rejected.append({"field": field, "reason": reason})
    selected = _preferred(valid, cfg.get("preferred_order", [])) or (valid[0] if valid else None)
    blocking_errors = []
    if selected is None:
        blocking_errors.append("no valid RLDS image tensor field was resolved")
    return {
        "image_field": selected,
        "image_field_valid": selected is not None,
        "image_field_is_metadata_flag": _is_metadata_image_flag(selected),
        "image_candidate_fields": candidates,
        "image_tensor_fields": valid,
        "image_flag_fields": [field for field in candidates if _is_metadata_image_flag(field)],
        "image_rejected_fields": rejected,
        "blocking_errors": blocking_errors,
        "warnings": [],
    }


def resolve_action_field(action_fields: list[str], policy: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**DEFAULT_POLICY["action"], **(policy or {})}
    candidates = sorted({str(field) for field in action_fields})
    selected = _preferred(candidates, cfg.get("preferred_order", [])) or (candidates[0] if candidates else None)
    return {
        "action_field": selected,
        "action_field_valid": selected is not None,
        "action_candidate_fields": candidates,
        "action_used_as_input": False,
        "warnings": [] if selected else ["no RLDS action field resolved"],
    }


def resolve_language_field(language_fields: list[str], policy: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**DEFAULT_POLICY["language"], **(policy or {})}
    candidates = sorted({str(field) for field in language_fields})
    selected = _preferred(candidates, cfg.get("preferred_order", [])) or (candidates[0] if candidates else None)
    language_is_text = bool(selected and "embedding" not in selected.lower())
    warnings = []
    if selected and not language_is_text:
        warnings.append("language field resolved to embedding because no text instruction field was available")
    if not selected:
        warnings.append("no RLDS language field resolved")
    return {
        "language_field": selected,
        "language_field_valid": selected is not None,
        "language_candidate_fields": candidates,
        "language_text_fields": [field for field in candidates if "embedding" not in field.lower()],
        "language_embedding_fields": [field for field in candidates if "embedding" in field.lower()],
        "language_is_text": language_is_text,
        "language_used_as_input": False,
        "warnings": warnings,
    }


def resolve_goal_field(goal_fields: list[str], policy: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**DEFAULT_POLICY["goal"], **(policy or {})}
    candidates = sorted({str(field) for field in goal_fields})
    selected = _preferred(candidates, cfg.get("preferred_order", [])) or (candidates[0] if candidates else None)
    return {
        "goal_field": selected,
        "goal_field_valid": selected is not None,
        "goal_candidate_fields": candidates,
        "goal_used_as_input": False,
        "warnings": [],
    }


def is_metadata_image_flag(field: str | None) -> bool:
    return _is_metadata_image_flag(field)


def _merge_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    if policy is None:
        return DEFAULT_POLICY
    merged = {}
    for key, value in DEFAULT_POLICY.items():
        merged[key] = {**value, **(policy.get(key, {}) if isinstance(policy.get(key, {}), dict) else {})}
    return merged


def _image_rejection_reason(field: str, cfg: dict[str, Any]) -> str | None:
    if _is_metadata_image_flag(field):
        return "metadata flag, not image tensor"
    for pattern in cfg.get("exclude_patterns", []):
        if pattern and pattern in field:
            return "excluded by image field policy"
    required_prefix = cfg.get("required_prefix")
    if required_prefix and not field.startswith(required_prefix):
        return f"does not start with required prefix {required_prefix}"
    if "image" not in field.lower():
        return "field name does not contain image"
    include_patterns = cfg.get("include_patterns", [])
    if include_patterns and not any(pattern in field for pattern in include_patterns):
        return "does not match image include patterns"
    return None


def _is_metadata_image_flag(field: str | None) -> bool:
    if field is None:
        return False
    return field.startswith("episode_metadata/") or "/has_image_" in field or field.endswith("has_image")


def _preferred(candidates: list[str], preferred_order: list[str]) -> str | None:
    candidate_set = set(candidates)
    for preferred in preferred_order:
        if preferred in candidate_set:
            return preferred
    return None
