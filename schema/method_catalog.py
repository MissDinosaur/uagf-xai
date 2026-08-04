"""Canonical catalogue for evidence planning, normalization, and reporting.

The catalogue is intentionally declarative. It describes the methods that are
actually implemented in UAGF-XAI; it is not a list of aspirational thesis
features.
"""

from __future__ import annotations

from dataclasses import dataclass

from layers.llm.evidence_methods import (
    LLM_GROUNDING,
    LLM_PROMPT_FAIRNESS,
    LLM_SELF_CONSISTENCY,
    LLM_SEMANTIC_DRIFT,
)


CATALOG_VERSION = "1.0"

CLASSIFICATION_TASKS = frozenset(
    {"binary_classification", "multiclass_classification"}
)
REGRESSION_TASKS = frozenset({"regression", "forecasting"})


@dataclass(frozen=True)
class MethodDefinition:
    """One evidence method and the constraints relevant to its execution."""

    token: str
    result_key: str
    evidence_id: str
    layer: str
    method: str
    evidence_type: str
    display_name: str
    short_name: str
    executive_item: str
    articles: tuple[str, ...]
    compatible_tasks: frozenset[str]
    requirements: tuple[str, ...]
    pathway: str

    @property
    def report_layer(self) -> str:
        labels = {
            "explainability": "Explainability",
            "fairness": "Fairness",
            "uncertainty": "Uncertainty",
            "drift": "Drift",
            "llm_explainability": "LLM Grounding",
            "llm_uncertainty": "LLM Uncertainty",
            "llm_drift": "LLM Drift",
            "llm_fairness": "LLM Fairness",
        }
        return labels[self.layer]

    @property
    def report_evidence_type(self) -> str:
        labels = {
            "feature_attribution": "Feature attribution",
            "local_surrogate_explanation": "Local surrogate explanation",
            "counterfactual_explanation": "Counterfactual explanation",
            "group_fairness_metrics": "Group fairness metrics",
            "conformal_prediction": "Conformal prediction",
            "dataset_feature_drift": "Dataset / feature drift",
            "grounding_evaluation": "Grounding evaluation",
            "self_consistency_evaluation": "Self-consistency evaluation",
            "semantic_drift_evaluation": "Semantic drift evaluation",
            "differential_prompt_fairness": "Differential prompt fairness",
        }
        return labels[self.evidence_type]


METHOD_CATALOG: dict[str, MethodDefinition] = {
    "shap": MethodDefinition(
        token="shap",
        result_key="explainability",
        evidence_id="EXP-SHAP",
        layer="explainability",
        method="SHAP",
        evidence_type="feature_attribution",
        display_name="Explainability Evidence \u2014 SHAP Feature Attribution",
        short_name="SHAP",
        executive_item="SHAP: Feature Attribution",
        articles=("Art. 13",),
        compatible_tasks=frozenset(
            {
                "binary_classification",
                "multiclass_classification",
                "regression",
                "forecasting",
                "anomaly_detection",
            }
        ),
        requirements=("predict", "evaluation features"),
        pathway="traditional",
    ),
    "lime": MethodDefinition(
        token="lime",
        result_key="lime",
        evidence_id="EXP-LIME",
        layer="explainability",
        method="LIME",
        evidence_type="local_surrogate_explanation",
        display_name="Explainability Evidence \u2014 LIME Local Explanation",
        short_name="LIME",
        executive_item="LIME: Local Explanation",
        articles=("Art. 13",),
        compatible_tasks=CLASSIFICATION_TASKS | frozenset({"regression"}),
        requirements=("predict or predict_proba", "numeric model-ready features"),
        pathway="traditional",
    ),
    "dice": MethodDefinition(
        token="dice",
        result_key="counterfactual",
        evidence_id="EXP-DICE",
        layer="explainability",
        method="DiCE",
        evidence_type="counterfactual_explanation",
        display_name="Explainability Evidence \u2014 DiCE Counterfactual Explanation",
        short_name="DiCE",
        executive_item="DiCE: Counterfactual Explanation",
        articles=("Art. 13",),
        compatible_tasks=CLASSIFICATION_TASKS,
        requirements=("predict_proba", "evaluation labels"),
        pathway="traditional",
    ),
    "fairness": MethodDefinition(
        token="fairness",
        result_key="fairness",
        evidence_id="FAIR-FAIRLEARN",
        layer="fairness",
        method="Fairlearn",
        evidence_type="group_fairness_metrics",
        display_name="Fairness Evidence \u2014 Fairlearn",
        short_name="Fairlearn",
        executive_item="Fairlearn",
        articles=("Art. 10",),
        compatible_tasks=CLASSIFICATION_TASKS,
        requirements=("predict", "evaluation labels", "S5 sensitive features"),
        pathway="traditional",
    ),
    "uncertainty": MethodDefinition(
        token="uncertainty",
        result_key="uncertainty",
        evidence_id="UNC-MAPIE",
        layer="uncertainty",
        method="MAPIE (Conformal Prediction)",
        evidence_type="conformal_prediction",
        display_name="Uncertainty Evidence \u2014 MAPIE",
        short_name="MAPIE",
        executive_item="MAPIE",
        articles=("Art. 9", "Art. 14", "Art. 15"),
        compatible_tasks=CLASSIFICATION_TASKS | REGRESSION_TASKS,
        requirements=("fitted sklearn-compatible estimator", "evaluation labels"),
        pathway="traditional",
    ),
    "drift": MethodDefinition(
        token="drift",
        result_key="drift",
        evidence_id="DRIFT-EVIDENTLY",
        layer="drift",
        method="Evidently + Feature Drift Tests",
        evidence_type="dataset_feature_drift",
        display_name="Drift Evidence \u2014 Evidently + Feature Drift Tests",
        short_name="Evidently",
        executive_item="Evidently + Feature Drift Tests",
        articles=("Art. 15", "Art. 61"),
        compatible_tasks=frozenset(
            {
                "binary_classification",
                "multiclass_classification",
                "regression",
                "forecasting",
                "anomaly_detection",
            }
        ),
        requirements=("reference data", "current data"),
        pathway="traditional",
    ),
    LLM_GROUNDING: MethodDefinition(
        token=LLM_GROUNDING,
        result_key=LLM_GROUNDING,
        evidence_id="LLM-E1",
        layer="llm_explainability",
        method="LLM-E1 Grounding Score",
        evidence_type="grounding_evaluation",
        display_name="LLM-E1 \u2014 Grounding Score",
        short_name="LLM-E1 Grounding Score",
        executive_item="LLM-E1 \u2014 Grounding Score",
        articles=("Art. 13",),
        compatible_tasks=frozenset({"llm_generation"}),
        requirements=("loadable LLM", "golden set"),
        pathway="llm",
    ),
    LLM_SELF_CONSISTENCY: MethodDefinition(
        token=LLM_SELF_CONSISTENCY,
        result_key=LLM_SELF_CONSISTENCY,
        evidence_id="LLM-E2",
        layer="llm_uncertainty",
        method="LLM-E2 Self-Consistency Score",
        evidence_type="self_consistency_evaluation",
        display_name="LLM-E2 \u2014 Self-Consistency Score",
        short_name="LLM-E2 Self-Consistency Score",
        executive_item="LLM-E2 \u2014 Self-Consistency Score",
        articles=("Art. 9", "Art. 14", "Art. 15"),
        compatible_tasks=frozenset({"llm_generation"}),
        requirements=("loadable LLM", "golden set", "repeatable generation"),
        pathway="llm",
    ),
    LLM_SEMANTIC_DRIFT: MethodDefinition(
        token=LLM_SEMANTIC_DRIFT,
        result_key=LLM_SEMANTIC_DRIFT,
        evidence_id="LLM-E3",
        layer="llm_drift",
        method="LLM-E3 Semantic Drift Index",
        evidence_type="semantic_drift_evaluation",
        display_name="LLM-E3 \u2014 Semantic Drift Index",
        short_name="LLM-E3 Semantic Drift Index",
        executive_item="LLM-E3 \u2014 Semantic Drift Index",
        articles=("Art. 15", "Art. 61"),
        compatible_tasks=frozenset({"llm_generation"}),
        requirements=("loadable LLM", "reference prompts", "current prompts"),
        pathway="llm",
    ),
    LLM_PROMPT_FAIRNESS: MethodDefinition(
        token=LLM_PROMPT_FAIRNESS,
        result_key=LLM_PROMPT_FAIRNESS,
        evidence_id="LLM-E4",
        layer="llm_fairness",
        method="LLM-E4 Differential Prompt Fairness",
        evidence_type="differential_prompt_fairness",
        display_name="LLM-E4 \u2014 Differential Prompt Fairness",
        short_name="LLM-E4 Differential Prompt Fairness",
        executive_item="LLM-E4 \u2014 Differential Prompt Fairness",
        articles=("Art. 10",),
        compatible_tasks=frozenset({"llm_generation"}),
        requirements=("loadable LLM", "paired fairness prompts"),
        pathway="llm",
    ),
}

TRADITIONAL_METHOD_TOKENS = tuple(
    token for token, item in METHOD_CATALOG.items() if item.pathway == "traditional"
)
LLM_METHOD_TOKENS = tuple(
    token for token, item in METHOD_CATALOG.items() if item.pathway == "llm"
)
METHOD_TASK_COMPATIBILITY = {
    token: set(item.compatible_tasks) for token, item in METHOD_CATALOG.items()
}
RESULT_KEY_TO_TOKEN = {
    item.result_key: token for token, item in METHOD_CATALOG.items()
}


def article_method_map(pathway: str) -> dict[str, list[str]]:
    """Build compact article keys used by S5 from canonical catalogue entries."""
    mapping: dict[str, list[str]] = {}
    tokens = (
        TRADITIONAL_METHOD_TOKENS if pathway == "traditional" else LLM_METHOD_TOKENS
    )
    for token in tokens:
        for article in METHOD_CATALOG[token].articles:
            compact = article.replace(". ", "")
            mapping.setdefault(compact, []).append(token)
    return mapping
