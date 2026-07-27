"""Central report-facing titles for evidence methods.

Internal planner tokens and evidence payload method names remain unchanged.
"""

from __future__ import annotations


REPORT_METHOD_TITLES = {
    "shap": "Explainability Evidence — SHAP Feature Attribution",
    "explainability": "Explainability Evidence — SHAP Feature Attribution",
    "lime": "Explainability Evidence — LIME Local Explanation",
    "dice": "Explainability Evidence — DiCE Counterfactual Explanation",
    "counterfactual": "Explainability Evidence — DiCE Counterfactual Explanation",
    "fairness": "Fairness Evidence — Fairlearn",
    "uncertainty": "Uncertainty Evidence — MAPIE",
    "drift": "Drift Evidence — Evidently + Feature Drift Tests",
    "llm_grounding": "LLM-E1 — Grounding Score",
    "llm_self_consistency": "LLM-E2 — Self-Consistency Score",
    "llm_semantic_drift": "LLM-E3 — Semantic Drift Index",
    "llm_prompt_fairness": "LLM-E4 — Differential Prompt Fairness",
}


def report_method_title(method_token: str, fallback: str | None = None) -> str:
    """Return a polished title without changing the internal method token."""
    return REPORT_METHOD_TITLES.get(method_token, fallback or method_token)
