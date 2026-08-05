from adapters.s4_governance_adapter import GovernanceContext
from adapters.s5_audit_adapter import AuditContext
from layers.llm.llm_evidence_methods import LLM_METHOD_ORDER
from planner.cbep import plan_evidence


def _audit(
    task_type,
    *,
    system_type="traditional_ml",
    modality=None,
    articles=None,
):
    return AuditContext(
        system_type=system_type,
        modality=modality
        or ("text" if system_type in {"llm", "agentic"} else "tabular"),
        application_domain="test",
        risk_tier="high",
        applicable_articles=list(articles or []),
        blocking_findings=[],
        csp_satisfied=True,
        sensitive_feature_columns=["gender"],
        task_type=task_type,
    )


def test_high_risk_binary_classification_selects_traditional_evidence():
    methods, trace = plan_evidence(
        _audit("binary_classification", articles=["Art10", "Art13", "Art15"])
    )

    assert methods == ["shap", "lime", "dice", "fairness", "uncertainty", "drift"]
    assert trace["final_plan"] == methods
    assert trace["planning_model"] == "deterministic_constraint_informed"


def test_forecasting_adds_supported_uncertainty_and_filters_classification_only_methods():
    methods, trace = plan_evidence(_audit("forecasting", articles=["Art13"]))

    assert methods == ["shap", "uncertainty", "drift"]
    incompatible = trace["task_compatibility_assessment"]["incompatible_methods"]
    assert incompatible == ["fairness", "lime", "dice"]
    assert "uncertainty" not in incompatible


def test_anomaly_detection_filters_probability_dependent_methods():
    methods, trace = plan_evidence(_audit("anomaly_detection", articles=["Art13"]))

    assert methods == ["shap", "drift"]
    incompatible = trace["task_compatibility_assessment"]["incompatible_methods"]
    assert "dice" in incompatible
    assert "uncertainty" in incompatible
    decisions = {item["method"]: item for item in trace["method_decisions"]}
    assert decisions["uncertainty"]["status"] == "incompatible_filtered"
    assert "anomaly_detection" in decisions["uncertainty"]["reason"]


def test_high_risk_agentic_case_selects_professor_defined_llm_path():
    methods, trace = plan_evidence(
        _audit("llm_generation", system_type="agentic", articles=[])
    )

    assert methods == LLM_METHOD_ORDER
    assert trace["final_plan_display_names"] == [
        "LLM-E1 Grounding Score",
        "LLM-E2 Self-Consistency Score",
        "LLM-E3 Semantic Drift Index",
        "LLM-E4 Differential Prompt Fairness",
    ]


def test_trace_contains_planning_and_governance_inputs():
    governance = GovernanceContext(
        governance_score=3.0,
        governance_verdict="PASS_WITH_OBSERVATIONS",
        domain_scores={"Monitoring and Incident Response": 2.0},
    )

    _, trace = plan_evidence(_audit("binary_classification"), governance)

    for key in (
        "base_plan",
        "article_plan",
        "final_plan",
        "task_compatibility_assessment",
        "governance_context",
    ):
        assert key in trace
    assert trace["governance_context"]["status"] == "applied"


def test_text_classification_filters_dice_before_execution():
    methods, trace = plan_evidence(
        _audit(
            "binary_classification",
            modality="text",
            articles=["Art10", "Art13", "Art15"],
        )
    )

    assert methods == ["shap", "lime", "fairness", "uncertainty", "drift"]
    assessment = trace["task_compatibility_assessment"]
    assert assessment["normalized_modality"] == "text"
    assert assessment["incompatible_methods"] == ["dice"]
    assert "tabular counterfactuals" in assessment["incompatibility_reasons"]["dice"]


def test_unsupported_traditional_modalities_are_explicitly_filtered():
    for modality in ("image", "audio"):
        methods, trace = plan_evidence(
            _audit(
                "binary_classification",
                modality=modality,
                articles=["Art10", "Art13", "Art15"],
            )
        )

        assert methods == []
        assessment = trace["task_compatibility_assessment"]
        assert assessment["incompatible_methods"]
        assert all(
            modality in reason
            for reason in assessment["incompatibility_reasons"].values()
        )
