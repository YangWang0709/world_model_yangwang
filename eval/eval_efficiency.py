"""Efficiency evaluation placeholder."""


def token_retention_ratio(selected_tokens: int, total_tokens: int) -> float:
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    return float(selected_tokens) / float(total_tokens)

