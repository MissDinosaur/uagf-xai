from __future__ import annotations

from pathlib import Path

from adapters.s5_audit_adapter import AuditContext
from layers.llm.evidence_methods import LLM_METHOD_ORDER, llm_method_display_name
from report import report_generator
from report.method_titles import report_method_title
from schema.evidence_schema import not_applicable_evidence, skipped_evidence


def _trace(final_plan, incompatible=None):
    return {
        "type": "cbep_planning_trace",
        "method": "CBEP",
        "base_plan": list(final_plan),
        "article_plan": [],
        "final_plan": list(final_plan),
        "task_compatibility_assessment": {
            "incompatible_methods": list(incompatible or []),
        },
    }


def _output_to(tmp_path, monkeypatch, filename="report.html"):
    path = tmp_path / filename
    monkeypatch.setattr(report_generator, "build_output_path", lambda *args, **kwargs: str(path))
    return path


def test_html_report_contains_professional_sections_and_raw_evidence(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
    unified_result_factory,
):
    output = _output_to(tmp_path, monkeypatch)
    results = {
        "explainability": unified_result_factory(),
        "_cbep_trace": _trace(["shap"]),
    }

    returned = report_generator.generate_report(
        results,
        "high",
        provider_name="Test Provider",
        audit_context=traditional_audit_context,
        governance_context=governance_context,
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")

    assert returned == str(output)
    for heading in (
        "Executive Summary",
        "Runtime and Reproducibility Information",
        "Python version",
        "Runtime per method",
        "Output Artifacts Generated",
        "CBEP Evidence Planning Summary",
        "Evidence Coverage Matrix",
        "Evidence Findings by Layer",
        "Raw Evidence Appendix",
    ):
        assert heading in html
    assert "raw-marker" in html
    assert "status-completed" in html
    assert "<h3>Package versions</h3>" not in html
    assert '<section id="limitations">' not in html


def test_report_renders_skipped_and_not_applicable_statuses(
    tmp_path, monkeypatch, traditional_audit_context, governance_context
):
    output = _output_to(tmp_path, monkeypatch)
    results = {
        "fairness": skipped_evidence(
            evidence_id="FAIR-FAIRLEARN",
            layer="fairness",
            method="Fairlearn",
            summary="Skipped for test.",
            limitations=["No sensitive features."],
        ),
        "counterfactual": not_applicable_evidence(
            evidence_id="EXP-DICE",
            layer="counterfactual",
            method="DiCE",
            summary="Not applicable for test.",
            limitations=["Incompatible task."],
        ),
        "_cbep_trace": _trace(["fairness"], incompatible=["dice"]),
    }

    report_generator.generate_report(
        results,
        "high",
        audit_context=traditional_audit_context,
        governance_context=governance_context,
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")

    assert "status-skipped" in html
    assert "status-not-applicable" in html
    assert "No sensitive features." in html


def test_llm_report_uses_professor_defined_method_names(tmp_path, monkeypatch, governance_context):
    output = _output_to(tmp_path, monkeypatch)
    audit_context = AuditContext(
        system_type="agentic",
        modality="text",
        application_domain="justice",
        risk_tier="high",
        task_type="llm_generation",
        provider_name="LLM Provider",
    )
    results = {"_cbep_trace": _trace(LLM_METHOD_ORDER)}
    for index, token in enumerate(LLM_METHOD_ORDER, 1):
        results[token] = skipped_evidence(
            evidence_id=f"LLM-E{index}",
            layer=f"llm_layer_{index}",
            method=llm_method_display_name(token),
            summary="Metadata-only test result.",
            limitations=["No model weights."],
        )

    report_generator.generate_report(
        results,
        "high",
        audit_context=audit_context,
        governance_context=governance_context,
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")

    for token in LLM_METHOD_ORDER:
        assert report_method_title(token) in html


def test_llm_metadata_only_report_uses_explicit_resource_badge(
    tmp_path, monkeypatch, governance_context
):
    output = _output_to(tmp_path, monkeypatch)
    audit_context = AuditContext(
        system_type="agentic",
        modality="text",
        application_domain="justice",
        risk_tier="high",
        task_type="llm_generation",
        provider_name="LLM Provider",
        golden_set_uri="file://golden.json",
    )
    results = {"_cbep_trace": _trace(LLM_METHOD_ORDER)}
    for index, token in enumerate(LLM_METHOD_ORDER, 1):
        results[token] = skipped_evidence(
            evidence_id=f"LLM-E{index}",
            layer=f"llm_layer_{index}",
            method=llm_method_display_name(token),
            summary="Metadata-only test result.",
            limitations=["No model weights."],
        )

    report_generator.generate_report(
        results,
        "high",
        audit_context=audit_context,
        governance_context=governance_context,
        resource_context={
            "model_status": "metadata_only",
            "model_is_loadable": False,
            "golden_set": [{"prompt": "test"}],
            "golden_set_loaded": True,
        },
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")

    assert "status-metadata-only" in html
    assert "Metadata-only" in html
    assert "Golden-set records" in html
    assert ">1<" in html
    assert "agentic LLM system in the justice domain" in html


def test_generate_report_skips_pdf_export_when_disabled(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
    unified_result_factory,
):
    _output_to(tmp_path, monkeypatch)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("PDF exporter should not be called")

    monkeypatch.setattr(report_generator, "export_html_to_pdf", fail_if_called)
    report_generator.generate_report(
        {"explainability": unified_result_factory(), "_cbep_trace": _trace(["shap"])},
        "high",
        audit_context=traditional_audit_context,
        governance_context=governance_context,
        generate_pdf=False,
    )


def test_generate_report_invokes_pdf_exporter_when_enabled(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
    unified_result_factory,
):
    output = _output_to(tmp_path, monkeypatch)
    calls = []

    def fake_export(html_path, pdf_path):
        calls.append((Path(html_path), Path(pdf_path)))
        Path(pdf_path).write_bytes(b"%PDF-test")
        return Path(pdf_path).resolve()

    monkeypatch.setattr(report_generator, "export_html_to_pdf", fake_export)
    report_generator.generate_report(
        {"explainability": unified_result_factory(), "_cbep_trace": _trace(["shap"])},
        "high",
        audit_context=traditional_audit_context,
        governance_context=governance_context,
        generate_pdf=True,
    )

    assert calls == [(output, output.with_suffix(".pdf"))]
    assert output.with_suffix(".pdf").read_bytes() == b"%PDF-test"
