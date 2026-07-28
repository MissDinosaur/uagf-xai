from __future__ import annotations

from pathlib import Path

from adapters.s5_audit_adapter import AuditContext
from layers.llm.evidence_methods import LLM_METHOD_ORDER, llm_method_display_name
from report import report_generator
from report.method_titles import report_method_title
from schema.evidence_schema import not_applicable_evidence, skipped_evidence
from schema.evidence_schema import completed_evidence


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
    assert "Method Applicability and Skipped Evidence" not in html
    assert '<section id="method-applicability">' not in html
    assert "8. Raw Evidence Appendix" in html
    assert (
        "CBEP creates an article- and risk-driven plan and then applies task "
        "compatibility screening. Incompatible methods are excluded to avoid "
        "misleading or invalid evidence."
    ) in html


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
    assert 'class="callout limitation"' in html


def test_report_reorganizes_scope_resources_and_runtime_without_duplicates(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
    unified_result_factory,
):
    output = _output_to(tmp_path, monkeypatch)
    traditional_audit_context.model_artifact_uri = "file://data/model.joblib"
    traditional_audit_context.model_format = "joblib"
    traditional_audit_context.model_framework = "sklearn"
    traditional_audit_context.model_type = "classifier"
    traditional_audit_context.training_dataset_uri = "file://data/train.csv"
    traditional_audit_context.evaluation_dataset_uri = "file://data/eval.csv"

    report_generator.generate_report(
        {
            "explainability": unified_result_factory(),
            "_cbep_trace": _trace(["shap"]),
        },
        "high",
        audit_context=traditional_audit_context,
        governance_context=governance_context,
        resource_context={
            "resolved_path": r"data\case\model.joblib",
            "model_is_loadable": True,
            "training_dataset_loaded": True,
            "evaluation_dataset_loaded": True,
            "training_dataset_rows": 100,
            "evaluation_dataset_rows": 25,
        },
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")
    audit_scope = html.split('<section id="audit-scope">', 1)[1].split("</section>", 1)[0]
    resources = html.split('<section id="resource-loading">', 1)[1].split("</section>", 1)[0]
    runtime = html.split('<section id="runtime-reproducibility">', 1)[1].split("</section>", 1)[0]

    assert audit_scope.index("Contract-specific inputs") < audit_scope.index("Target column")
    assert audit_scope.count("Target column") == 1
    assert audit_scope.count("Positive label") == 1
    assert audit_scope.count("Sensitive features") == 1
    assert "Model and Traditional ML dataset contract" in resources
    assert "data/case/model.joblib" in resources
    assert "Training dataset URI" in resources
    assert "Training rows" in resources
    assert "Selected methods" not in runtime
    assert "Completed methods" not in runtime
    assert "Skipped / not applicable" not in runtime
    assert "Model and Traditional ML dataset contract" not in runtime
    assert 'class="callout limitation"><strong>Runtime measurement scope:' in runtime


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


def test_report_renders_structured_dice_evidence_under_explainability(
    tmp_path, monkeypatch, traditional_audit_context, governance_context
):
    output = _output_to(tmp_path, monkeypatch)
    result = completed_evidence(
        evidence_id="EXP-DICE",
        layer="explainability",
        method="DiCE",
        article_mapping=["Art. 13"],
        summary="Structured counterfactual explanation.",
        key_findings=["One counterfactual was generated."],
        metrics={
            "counterfactuals_count": 1,
            "original_prediction": 1,
            "desired_class": "opposite",
            "counterfactual_prediction": 0,
            "changed_features_count": 1,
            "changed_features": [
                {
                    "feature": "income",
                    "original_value": 80_000,
                    "counterfactual_value": 40_000,
                    "delta": -40_000,
                }
            ],
            "counterfactuals": [
                {
                    "counterfactual_id": 1,
                    "counterfactual_prediction": 0,
                    "number_of_changed_features": 1,
                    "changed_features": [{"feature": "income"}],
                },
                {
                    "counterfactual_id": 2,
                    "counterfactual_prediction": 0,
                    "number_of_changed_features": 2,
                    "changed_features": [
                        {"feature": "income"},
                        {"feature": "duration"},
                    ],
                },
            ],
        },
        artifacts=["outputs/dice/test_counterfactuals.json"],
        limitations=["Domain review is required."],
        raw_output={"counterfactuals": [{"changed_features": []}]},
    )

    report_generator.generate_report(
        {
            "counterfactual": result,
            "_cbep_trace": _trace(["dice"]),
        },
        "high",
        audit_context=traditional_audit_context,
        governance_context=governance_context,
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")

    assert "Explainability Evidence — DiCE Counterfactual Explanation" in html
    assert "Explainability · Counterfactual explanation · Art. 13" in html
    assert "Original Prediction" in html
    assert "Desired Class" in html
    assert "Changed Features Count" in html
    assert "1 (label meaning unavailable)" in html
    assert "0 (label meaning unavailable)" in html
    assert (
        "The table below shows the first generated counterfactual. The full "
        "structured JSON artifact contains all generated counterfactuals."
    ) in html
    assert "<th>Feature</th>" in html
    assert "<th>Original value</th>" in html
    assert "<th>Counterfactual value</th>" in html
    assert "All generated counterfactuals" in html
    assert "<th>Counterfactual ID</th>" in html
    assert "<th>Prediction</th>" in html
    assert "<th>Changed features count</th>" in html
    assert "<th>Changed feature names</th>" in html
    assert "income, duration" in html
    assert "Open full structured DiCE JSON artifact" in html
    assert "Counterfactual · Counterfactual explanation" not in html


def test_report_uses_explicit_prediction_label_mapping_when_available(
    tmp_path, monkeypatch, traditional_audit_context, governance_context
):
    output = _output_to(tmp_path, monkeypatch)
    traditional_audit_context.prediction_label_mapping = {
        0: "good credit risk",
        1: "bad credit risk",
    }
    result = completed_evidence(
        evidence_id="EXP-DICE",
        layer="explainability",
        method="DiCE",
        article_mapping=["Art. 13"],
        summary="Structured counterfactual explanation.",
        metrics={
            "original_prediction": 1,
            "counterfactual_prediction": 0,
            "counterfactuals": [
                {
                    "counterfactual_id": 1,
                    "counterfactual_prediction": 0,
                    "changed_features": [],
                }
            ],
        },
    )

    report_generator.generate_report(
        {"counterfactual": result, "_cbep_trace": _trace(["dice"])},
        "high",
        audit_context=traditional_audit_context,
        governance_context=governance_context,
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")

    assert "1 (bad credit risk)" in html
    assert "0 (good credit risk)" in html
