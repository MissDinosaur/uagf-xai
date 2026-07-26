"""Canonical identifiers and professor-defined names for LLM evidence."""

LLM_GROUNDING = "llm_grounding"
LLM_SELF_CONSISTENCY = "llm_self_consistency"
LLM_SEMANTIC_DRIFT = "llm_semantic_drift"
LLM_PROMPT_FAIRNESS = "llm_prompt_fairness"

LLM_E1_GROUNDING_SCORE = "LLM-E1 Grounding Score"
LLM_E2_SELF_CONSISTENCY_SCORE = "LLM-E2 Self-Consistency Score"
LLM_E3_SEMANTIC_DRIFT_INDEX = "LLM-E3 Semantic Drift Index"
LLM_E4_DIFFERENTIAL_PROMPT_FAIRNESS = "LLM-E4 Differential Prompt Fairness"

LLM_METHOD_DISPLAY_NAMES = {
    LLM_GROUNDING: LLM_E1_GROUNDING_SCORE,
    LLM_SELF_CONSISTENCY: LLM_E2_SELF_CONSISTENCY_SCORE,
    LLM_SEMANTIC_DRIFT: LLM_E3_SEMANTIC_DRIFT_INDEX,
    LLM_PROMPT_FAIRNESS: LLM_E4_DIFFERENTIAL_PROMPT_FAIRNESS,
}

# Temporary aliases make existing stored traces and callers readable while the
# canonical CBEP tokens use the professor-defined evidence concepts.
LLM_METHOD_DISPLAY_NAMES.update(
    {
        "llm_explainability": LLM_E1_GROUNDING_SCORE,
        "llm_uncertainty": LLM_E2_SELF_CONSISTENCY_SCORE,
        "llm_drift": LLM_E3_SEMANTIC_DRIFT_INDEX,
        "llm_fairness": LLM_E4_DIFFERENTIAL_PROMPT_FAIRNESS,
    }
)

LLM_METHOD_ORDER = [
    LLM_GROUNDING,
    LLM_SELF_CONSISTENCY,
    LLM_SEMANTIC_DRIFT,
    LLM_PROMPT_FAIRNESS,
]


def llm_method_display_name(method_token: str) -> str:
    """Return the professor-defined report name for an LLM method token."""
    return LLM_METHOD_DISPLAY_NAMES.get(method_token, method_token)
