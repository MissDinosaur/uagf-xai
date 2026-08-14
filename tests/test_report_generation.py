from __future__ import annotations

from pathlib import Path

from adapters.s5_audit_adapter import AuditContext
from layers.llm.llm_evidence_methods import LLM_METHOD_ORDER, llm_method_display_name
from report import report_generator
from report.report_builder import build_cbep_decision_table
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
    assert "deterministic, constraint-informed planning procedure" in html
    assert "Incompatible methods are excluded" in html
    coverage = html.split('<section id="coverage-matrix">', 1)[1].split(
        "</section>", 1
    )[0]
    assert "EU AI Act Article" not in coverage


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
    findings = html.split('<section id="evidence-findings">', 1)[1].split(
        "</section>", 1
    )[0]
    raw_evidence = html.split('<section id="raw-evidence">', 1)[1].split(
        "</section>", 1
    )[0]
    assert "Skipped for test." in findings
    assert "Not applicable for test." not in findings
    assert "Raw Fairness Evidence" in raw_evidence
    assert "Raw Explainability Evidence â€” DiCE" not in raw_evidence
    assert "Not applicable for test." not in raw_evidence


def test_anomaly_report_explains_task_compatibility_filtering():
    audit_context = AuditContext(
        system_type="traditional_ml",
        modality="tabular",
        application_domain="critical_infrastructure",
        risk_tier="high",
        task_type="anomaly_detection",
    )
    results = {
        "_cbep_trace": _trace(
            ["shap", "drift"],
            incompatible=["fairness", "uncertainty", "lime", "dice"],
        )
    }

    table = build_cbep_decision_table(results, audit_context)

    assert "evaluated the traditional evidence library" in table["intro"]
    assert "reported as non-applicable" in table["intro"]
    rows = {row["method"]: row for row in table["rows"]}
    assert rows["MAPIE"]["selected"] == "No"
    assert "IsolationForest" in rows["MAPIE"]["reason"]


def test_cbep_table_uses_modality_reason_from_planning_trace():
    audit_context = AuditContext(
        system_type="traditional_ml",
        modality="text",
        risk_tier="high",
        task_type="binary_classification",
    )
    results = {
        "_cbep_trace": _trace(
            ["shap", "lime", "fairness", "uncertainty", "drift"],
            incompatible=["dice"],
        )
    }
    reason = (
        "Excluded because the current DiCE implementation supports tabular "
        "counterfactuals and is not compatible with text modality."
    )
    results["_cbep_trace"]["task_compatibility_assessment"][
        "incompatibility_reasons"
    ] = {"dice": reason}

    table = build_cbep_decision_table(results, audit_context)

    rows = {row["method"]: row for row in table["rows"]}
    assert rows["DiCE"]["reason"] == reason
    assert "task and modality compatibility screening" in table["intro"]


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
    assert "Task type" not in audit_scope
    assert "Model feature columns" not in audit_scope
    assert "Model and Traditional ML dataset contract" in resources
    assert "data/case/model.joblib" in resources
    assert "Training dataset URI" in resources
    assert "Training rows" in resources
    for text_only_label in (
        "Model input adapter",
        "Text feature column",
        "TF-IDF vectorizer",
        "Final estimator",
        "TF-IDF vocabulary size",
    ):
        assert text_only_label not in resources
    assert "Selected methods" not in runtime
    assert "Completed methods" not in runtime
    assert "Skipped / not applicable" not in runtime
    assert "Model and Traditional ML dataset contract" not in runtime
    assert 'class="callout limitation"><strong>Runtime measurement scope:' in runtime


def test_executive_summary_uses_separate_system_and_task_cards(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
    unified_result_factory,
):
    output = _output_to(tmp_path, monkeypatch)
    report_generator.generate_report(
        {
            "explainability": unified_result_factory(),
            "_cbep_trace": _trace(["shap"]),
        },
        "high",
        audit_context=traditional_audit_context,
        governance_context=governance_context,
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")
    executive = html.split('<section id="executive-summary">', 1)[1].split(
        "</section>", 1
    )[0]

    assert "System type / Modality" in executive
    assert "Task type" in executive
    assert "System / task" not in executive
    assert "Governance verdict" not in executive


def test_text_reports_include_text_specific_contract_metadata(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
    unified_result_factory,
):
    output = _output_to(tmp_path, monkeypatch)
    traditional_audit_context.modality = "text"
    traditional_audit_context.feature_columns = ["cv_text"]
    report_generator.generate_report(
        {
            "explainability": unified_result_factory(),
            "_cbep_trace": _trace(["shap"]),
        },
        "high",
        audit_context=traditional_audit_context,
        governance_context=governance_context,
        resource_context={
            "input_adapter": "sklearn_text",
            "text_feature_column": "cv_text",
            "vectorizer_class": "TfidfVectorizer",
            "estimator_class": "LogisticRegression",
            "vocabulary_size": 742,
        },
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")
    audit_scope = html.split('<section id="audit-scope">', 1)[1].split(
        "</section>", 1
    )[0]
    resources = html.split('<section id="resource-loading">', 1)[1].split(
        "</section>", 1
    )[0]

    assert "Model feature columns" in audit_scope
    assert "cv_text" in audit_scope
    assert "Model input adapter" in resources
    assert "sklearn_text" in resources
    assert "TF-IDF vocabulary size" in resources
    assert ">742<" in resources


def test_talentsift_report_uses_text_drift_and_ngram_semantics(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
    unified_result_factory,
):
    output = _output_to(tmp_path, monkeypatch)
    traditional_audit_context.modality = "text"
    traditional_audit_context.feature_columns = ["cv_text"]
    shap = unified_result_factory(
        metrics={
            "feature_semantics": "tokens_and_ngrams",
            "top_features": ["with years"],
            "feature_importance": [
                {"feature": "with years", "importance": 0.041466}
            ],
        }
    )
    drift = unified_result_factory(
        evidence_id="DRIFT-EVIDENTLY",
        layer="drift",
        method="Evidently + Feature Drift Tests",
        metrics={
            "dataset_drift_detected": True,
            "drift_share": 0.75,
            "drifted_feature_count": 3,
            "features_analyzed": 4,
            "model_feature_columns": ["cv_text"],
        },
    )

    report_generator.generate_report(
        {
            "explainability": shap,
            "drift": drift,
            "_cbep_trace": _trace(["shap", "drift"]),
        },
        "high",
        provider_name="TalentSift GmbH",
        audit_context=traditional_audit_context,
        governance_context=governance_context,
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")

    expected = (
        "Canonical text-input drift was detected for the authoritative cv_text "
        "model input: three of four derived statistical characteristics were "
        "flagged as drifted, producing a drift share of 0.75."
    )
    assert expected in html
    assert "Canonical model-input drift was detected across 4" not in html
    assert "Feature Semantics</span><span class=\"fact-value\">" in html
    assert "tokens_and_ngrams" in html
    assert "<th>Token / n-gram</th>" in html


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
        positive_label=1,
        feature_columns=["prompt"],
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
    audit_scope = html.split('<section id="audit-scope">', 1)[1].split(
        "</section>", 1
    )[0]
    assert "Legacy target-drop fallback" not in html
    assert "<th>Model feature columns</th><td>Not applicable</td>" in audit_scope
    assert "<th>Target column</th><td>Not applicable</td>" in audit_scope
    assert "<th>Positive label</th><td>Not applicable</td>" in audit_scope
    assert html.count("status-skipped") >= 4


def test_local_surrogate_report_separates_generator_and_embedding_model(
    tmp_path, monkeypatch, governance_context
):
    output = _output_to(tmp_path, monkeypatch)
    audit_context = AuditContext(
        system_type="llm",
        modality="text",
        task_type="llm_generation",
        risk_tier="high",
        provider_name="Local Surrogate",
        model_type="distilgpt2_local_surrogate",
    )

    report_generator.generate_report(
        {"_cbep_trace": _trace([])},
        "high",
        audit_context=audit_context,
        governance_context=governance_context,
        resource_context={
            "model_status": "loaded",
            "model_is_loadable": True,
            "model_container": "LocalCausalLMArtifact",
            "is_local_surrogate": True,
            "runtime_mode": "local",
            "evaluation_embedding_model_name": (
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "evaluation_embedding_metadata": {
                "resolved_path": "data/embedding/all-MiniLM-L6-v2",
                "pooling": "attention_mask_mean_pooling",
                "normalization": "l2",
                "local_files_only": True,
            },
        },
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")

    assert "Local surrogate execution validation" in html
    assert "resulting scores are not an audit of the original LegalMind model" in html
    assert "Audited/surrogate generation model" in html
    assert "DistilGPT2" in html
    assert "Evaluation embedding model" in html
    assert "sentence-transformers/all-MiniLM-L6-v2" in html
    assert "local high-risk justice-domain validation context" in html
    assert "Audit-context risk tier" in html
    assert "S5 risk tier" not in html
    assert "Validation context origin" not in html
    assert "Governance scenario" not in html
    assert "reused_legalmind_validation_scenario" not in html
    assert "s6_local_execution_fallback" not in html


def test_legacy_traditional_text_report_keeps_target_drop_fallback(
    tmp_path, monkeypatch, governance_context, unified_result_factory
):
    output = _output_to(tmp_path, monkeypatch)
    audit_context = AuditContext(
        system_type="traditional_ml",
        modality="text",
        task_type="binary_classification",
        feature_columns=None,
    )

    report_generator.generate_report(
        {"explainability": unified_result_factory(), "_cbep_trace": _trace(["shap"])},
        "high",
        audit_context=audit_context,
        governance_context=governance_context,
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")

    assert "<th>Model feature columns</th><td>Legacy target-drop fallback</td>" in html


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
            "counterfactual_policy_source": "audit_context_immutable_exclusions",
            "counterfactual_policy_status": "validated",
            "actionable_feature_columns": None,
            "immutable_feature_columns": ["age", "credit_history"],
            "excluded_sensitive_features": ["personal_status", "foreign_worker"],
            "excluded_immutable_features": ["age", "credit_history"],
            "excluded_non_actionable_features": [],
            "features_to_vary": ["income", "duration"],
            "policy_violation_detected": False,
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
        raw_output={
            "counterfactual_policy_source": "audit_context_immutable_exclusions",
            "counterfactual_policy_status": "validated",
            "excluded_sensitive_features": ["personal_status", "foreign_worker"],
            "excluded_immutable_features": ["age", "credit_history"],
            "features_to_vary": ["income", "duration"],
            "policy_violation_detected": False,
            "counterfactuals": [{"changed_features": []}],
        },
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
    assert "Counterfactual policy item" in html
    assert "Policy source" in html
    assert "audit_context_immutable_exclusions" in html
    assert "Policy validation status" in html
    assert "personal_status, foreign_worker" in html
    assert "age, credit_history" in html
    assert "income, duration" in html
    assert html.count("audit_context_immutable_exclusions") >= 2
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


def test_report_shows_abbreviated_hashes_and_raw_appendix_keeps_full_hashes(
    tmp_path, monkeypatch, traditional_audit_context, governance_context
):
    output = _output_to(tmp_path, monkeypatch)
    input_paths = {}
    for name in ("s4", "s5", "model", "train", "evaluation"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("utf-8"))
        input_paths[name] = path
    governance_context.source_json_path = input_paths["s4"]
    traditional_audit_context.source_json_path = input_paths["s5"]
    traditional_audit_context.model_artifact_uri = str(input_paths["model"])
    traditional_audit_context.training_dataset_uri = str(input_paths["train"])
    traditional_audit_context.evaluation_dataset_uri = str(
        input_paths["evaluation"]
    )

    report_generator.generate_report(
        {"_cbep_trace": _trace([])},
        "high",
        audit_context=traditional_audit_context,
        governance_context=governance_context,
        generate_pdf=False,
        runtime_context={
            "run_timestamp": "2026-08-06T12:30:00+02:00",
            "run_id": "fixed-run-id",
        },
    )
    html = output.read_text(encoding="utf-8")
    from report.runtime_metadata import sha256_file

    full_hash = sha256_file(input_paths["model"])
    runtime_section = html.split(
        '<section id="runtime-reproducibility">', 1
    )[1].split("</section>", 1)[0]
    raw_section = html.split('<section id="raw-evidence">', 1)[1]

    assert "Dependency Versions" in runtime_section
    assert "Input Provenance" in runtime_section
    assert f"{full_hash[:12]}…" in runtime_section
    assert full_hash not in runtime_section
    assert full_hash in raw_section
    assert "Raw Reproducibility metadata JSON" in raw_section
    assert "fixed-run-id" in html


def test_user_facing_html_excludes_internal_git_metadata(
    tmp_path, monkeypatch, traditional_audit_context, governance_context
):
    output = _output_to(tmp_path, monkeypatch)
    from report import runtime_metadata

    git_state = {
        "status": "available",
        "commit_sha": "deadbeef" * 5,
        "branch": "private-research-branch",
        "dirty": True,
    }
    monkeypatch.setattr(runtime_metadata, "collect_git_metadata", lambda root: git_state)

    report_generator.generate_report(
        {"_cbep_trace": _trace([])},
        "high",
        audit_context=traditional_audit_context,
        governance_context=governance_context,
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")

    for prohibited in (
        "Git commit",
        "Git branch",
        "Git dirty",
        "commit_sha",
        "private-research-branch",
        '"dirty": true',
    ):
        assert prohibited not in html
    assert "CBEP version" in html
    assert "Evidence schema / report version" in html
    assert "Dependency Versions" in html
    assert "Input Provenance" in html
    assert "SHA-256" in html


def test_harbour_report_labels_13_model_features_and_contextual_columns(
    tmp_path, monkeypatch, traditional_audit_context, governance_context
):
    output = _output_to(tmp_path, monkeypatch)
    model_features = [f"model_feature_{index}" for index in range(13)]
    drift = completed_evidence(
        evidence_id="DRIFT-EVIDENTLY",
        layer="drift",
        method="Evidently + Feature Drift Tests",
        article_mapping=["Art. 15", "Art. 61"],
        summary="Canonical model-input drift used 13 fitted model features.",
        key_findings=["13 model-input features were analyzed."],
        metrics={
            "feature_scope_source": "model_artifact_feature_cols",
            "model_feature_columns": model_features,
            "contextual_columns": ["crane_id", "visibility_m"],
            "contextual_dataset_drift": {
                "status": "supplementary_not_analyzed"
            },
            "dataset_drift_detected": False,
            "drift_share": 0.076923,
            "drifted_feature_count": 1,
            "features_analyzed": 13,
            "canonical_feature_tests": [],
        },
        raw_output={
            "feature_scope_source": "model_artifact_feature_cols",
            "model_feature_columns": model_features,
            "contextual_columns": ["crane_id", "visibility_m"],
            "features_analyzed": 13,
        },
    )

    report_generator.generate_report(
        {"drift": drift, "_cbep_trace": _trace(["drift"])},
        "high",
        provider_name="HarbourLogistik GmbH",
        audit_context=traditional_audit_context,
        governance_context=governance_context,
        generate_pdf=False,
    )
    html = output.read_text(encoding="utf-8")

    assert "Feature scope source" in html
    assert "model_artifact_feature_cols" in html
    assert "Supplementary contextual / excluded columns" in html
    assert "crane_id, visibility_m" in html
    assert "Model-input drift not detected; share 0.076923" in html
    assert html.count("model_feature_") >= 26
