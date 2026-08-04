from schema.method_catalog import (
    LLM_METHOD_TOKENS,
    METHOD_CATALOG,
    TRADITIONAL_METHOD_TOKENS,
)


def test_catalog_matches_the_implemented_six_plus_four_scope():
    assert TRADITIONAL_METHOD_TOKENS == (
        "shap",
        "lime",
        "dice",
        "fairness",
        "uncertainty",
        "drift",
    )
    assert LLM_METHOD_TOKENS == (
        "llm_grounding",
        "llm_self_consistency",
        "llm_semantic_drift",
        "llm_prompt_fairness",
    )
    assert len(METHOD_CATALOG) == 10


def test_catalog_contains_planning_and_report_metadata():
    for token, method in METHOD_CATALOG.items():
        assert method.token == token
        assert method.display_name
        assert method.layer
        assert method.evidence_type
        assert method.articles
        assert method.compatible_tasks
        assert method.requirements


def test_dice_is_explainability_counterfactual_evidence():
    method = METHOD_CATALOG["dice"]
    assert method.layer == "explainability"
    assert method.evidence_type == "counterfactual_explanation"


def test_forecasting_uncertainty_and_anomaly_compatibility_are_explicit():
    uncertainty = METHOD_CATALOG["uncertainty"]
    assert "forecasting" in uncertainty.compatible_tasks
    assert "anomaly_detection" not in uncertainty.compatible_tasks

