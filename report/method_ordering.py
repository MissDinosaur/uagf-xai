"""Central ordering rules for report-facing evidence methods and results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy

from schema.method_catalog import (
    LLM_METHOD_TOKENS,
    RESULT_KEY_TO_TOKEN,
    TRADITIONAL_METHOD_TOKENS,
)


TRADITIONAL_METHOD_ORDER = TRADITIONAL_METHOD_TOKENS
LLM_METHOD_ORDER = LLM_METHOD_TOKENS
METHOD_TO_RESULT_KEY = {
    token: result_key for result_key, token in RESULT_KEY_TO_TOKEN.items()
}


def is_llm_method_set(methods: Iterable[str]) -> bool:
    """Return whether a token collection belongs to the LLM evidence pathway."""
    return any(token in LLM_METHOD_ORDER for token in methods)


def ordered_method_tokens(methods: Iterable[str], *, is_llm: bool | None = None) -> list[str]:
    """Sort method tokens by the thesis layer order while preserving unknown tokens."""
    values = list(dict.fromkeys(methods))
    llm_path = is_llm_method_set(values) if is_llm is None else is_llm
    preferred = LLM_METHOD_ORDER if llm_path else TRADITIONAL_METHOD_ORDER
    rank = {token: index for index, token in enumerate(preferred)}
    original_rank = {token: index for index, token in enumerate(values)}
    return sorted(
        values,
        key=lambda token: (rank.get(token, len(rank)), original_rank[token]),
    )


def ordered_result_keys(results: Mapping[str, object]) -> list[str]:
    """Order report result keys consistently, with the CBEP trace kept first."""
    result_keys = list(results.keys())
    llm_path = any(token in results for token in LLM_METHOD_ORDER)
    method_order = LLM_METHOD_ORDER if llm_path else TRADITIONAL_METHOD_ORDER
    preferred_keys = [METHOD_TO_RESULT_KEY[token] for token in method_order]

    ordered = []
    if "_cbep_trace" in results:
        ordered.append("_cbep_trace")
    ordered.extend(key for key in preferred_keys if key in results)
    ordered.extend(key for key in result_keys if key not in ordered)
    return ordered


def ordered_cbep_trace(trace: Mapping[str, object]) -> dict:
    """Return a report-only CBEP trace copy with consistently ordered method lists."""
    ordered_trace = deepcopy(dict(trace))
    original_final_plan = list(ordered_trace.get("final_plan") or [])
    original_display_names = list(ordered_trace.get("final_plan_display_names") or [])
    display_by_token = dict(zip(original_final_plan, original_display_names))

    for field in ("base_plan", "article_plan", "final_plan"):
        values = ordered_trace.get(field)
        if isinstance(values, list):
            ordered_trace[field] = ordered_method_tokens(values)

    if original_display_names:
        ordered_trace["final_plan_display_names"] = [
            display_by_token.get(token, token)
            for token in ordered_trace.get("final_plan", [])
        ]

    compatibility = ordered_trace.get("task_compatibility_assessment")
    if isinstance(compatibility, dict):
        for field in ("compatible_methods", "incompatible_methods"):
            values = compatibility.get(field)
            if isinstance(values, list):
                compatibility[field] = ordered_method_tokens(values)

    return ordered_trace
