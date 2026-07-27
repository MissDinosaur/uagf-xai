from adapters.s4_governance_adapter import GovernanceAdapter


def test_governance_adapter_maps_overall_and_domain_scores(minimal_governance_json):
    context = GovernanceAdapter.from_cgsa_report(minimal_governance_json)

    assert context.governance_score == 3.4
    assert context.governance_verdict == "PASS_WITH_OBSERVATIONS"
    assert context.domain_scores["Transparency and Explainability"] == 3.1


def test_governance_adapter_fallback_is_neutral():
    context = GovernanceAdapter.fallback()

    assert context.governance_score == 5.0
    assert context.governance_verdict == "PASS"
    assert context.domain_scores == {}
