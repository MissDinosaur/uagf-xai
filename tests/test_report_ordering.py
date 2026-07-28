from __future__ import annotations

from html import unescape

from report import report_generator
from report.method_ordering import (
    LLM_METHOD_ORDER,
    TRADITIONAL_METHOD_ORDER,
    ordered_cbep_trace,
    ordered_method_tokens,
    ordered_result_keys,
)
from report.method_titles import report_method_title
from report.report_builder import METHOD_CATALOG
from schema.evidence_schema import completed_evidence


def _evidence(evidence_id, layer, method):
    return completed_evidence(
        evidence_id=evidence_id,
        layer=layer,
        method=method,
        article_mapping=[],
        summary=f"{method} summary.",
        key_findings=[],
        metrics={"runtime_seconds": 0.01},
        artifacts=[],
        limitations=[],
        raw_output={},
    )


def _assert_order(text, labels):
    positions = [text.index(label) for label in labels]
    assert positions == sorted(positions)


def test_traditional_method_order_follows_the_four_evidence_layers():
    shuffled = ["drift", "fairness", "dice", "shap", "uncertainty", "lime"]
    assert ordered_method_tokens(shuffled) == list(TRADITIONAL_METHOD_ORDER)


def test_dice_is_reported_as_explainability_counterfactual_evidence():
    assert METHOD_CATALOG["dice"]["layer"] == "Explainability"
    assert METHOD_CATALOG["dice"]["evidence_type"] == "Counterfactual explanation"
    assert report_method_title("dice").startswith("Explainability Evidence — DiCE")


def test_llm_method_order_follows_professor_defined_sequence():
    shuffled = [
        "llm_prompt_fairness",
        "llm_semantic_drift",
        "llm_grounding",
        "llm_self_consistency",
    ]
    assert ordered_method_tokens(shuffled) == list(LLM_METHOD_ORDER)


def test_report_trace_is_sorted_without_mutating_pipeline_results():
    trace = {
        "final_plan": ["drift", "shap", "dice", "lime"],
        "final_plan_display_names": ["Drift", "SHAP", "DiCE", "LIME"],
    }

    ordered = ordered_cbep_trace(trace)

    assert ordered["final_plan"] == ["shap", "lime", "dice", "drift"]
    assert ordered["final_plan_display_names"] == ["SHAP", "LIME", "DiCE", "Drift"]
    assert trace["final_plan"] == ["drift", "shap", "dice", "lime"]


def test_report_sections_use_one_central_order_for_shuffled_results(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
):
    output = tmp_path / "ordered-report.html"
    monkeypatch.setattr(
        report_generator,
        "build_output_path",
        lambda *args, **kwargs: str(output),
    )
    results = {
        "drift": _evidence("DRIFT-EVIDENTLY", "drift", "Evidently"),
        "fairness": _evidence("FAIR-FAIRLEARN", "fairness", "Fairlearn"),
        "counterfactual": _evidence("EXP-DICE", "counterfactual", "DiCE"),
        "lime": _evidence("EXP-LIME", "explainability", "LIME"),
        "uncertainty": _evidence("UNC-MAPIE", "uncertainty", "MAPIE"),
        "explainability": _evidence("EXP-SHAP", "explainability", "SHAP"),
        "_cbep_trace": {
            "final_plan": ["drift", "fairness", "dice", "shap", "uncertainty", "lime"],
            "task_compatibility_assessment": {"incompatible_methods": []},
        },
    }

    report_generator.generate_report(
        results,
        "high",
        audit_context=traditional_audit_context,
        governance_context=governance_context,
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")
    titles = [report_method_title(token) for token in TRADITIONAL_METHOD_ORDER]
    short_names = [METHOD_CATALOG[token]["short_name"] for token in TRADITIONAL_METHOD_ORDER]

    for section_id in ("runtime-reproducibility", "evidence-findings", "raw-evidence"):
        section = html.split(f'<section id="{section_id}">', 1)[1].split("</section>", 1)[0]
        _assert_order(section, titles)

    for section_id in ("cbep-planning", "coverage-matrix"):
        section = html.split(f'<section id="{section_id}">', 1)[1].split("</section>", 1)[0]
        _assert_order(section, short_names)

    executive = html.split('<section id="executive-summary">', 1)[1].split("</section>", 1)[0]
    selected_methods_box = executive.split("Selected methods", 1)[1].split(
        "Completed methods", 1
    )[0]
    _assert_order(
        selected_methods_box,
        [
            "SHAP: Feature Attribution",
            "LIME: Local Explanation",
            "DiCE: Counterfactual Explanation",
            "Fairlearn",
            "MAPIE",
            "Evidently + Feature Drift Tests",
        ],
    )
    assert "Explainability Evidence — SHAP, LIME and DiCE" in executive

    coverage = html.split('<section id="coverage-matrix">', 1)[1].split("</section>", 1)[0]
    assert (
        "<td>Explainability</td><td>Counterfactual explanation</td>"
        "<td>DiCE</td>"
    ) in coverage
    assert '<span class="method-group-title">Explainability Evidence</span>' in html
    assert "<ul class=\"method-group-items\"><li>" in html
    raw_section = unescape(html.split('<section id="raw-evidence">', 1)[1])
    _assert_order(
        raw_section,
        ['"shap"', '"lime"', '"dice"', '"fairness"', '"uncertainty"', '"drift"'],
    )
    assert ordered_result_keys(results) == [
        "_cbep_trace",
        "explainability",
        "lime",
        "counterfactual",
        "fairness",
        "uncertainty",
        "drift",
    ]
