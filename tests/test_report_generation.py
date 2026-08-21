from __future__ import annotations

from pathlib import Path

from adapters.s5_audit_adapter import AuditContext
from layers.llm.llm_evidence_methods import LLM_METHOD_ORDER, llm_method_display_name
from report import report_generator
from report.method_titles import report_method_title
from report.report_builder import build_cbep_decision_table
from schema.evidence_schema import completed_evidence, not_applicable_evidence
from schema.evidence_schema import skipped_evidence


def _trace(final_plan, incompatible=None, reasons=None):
    return {
        "type": "cbep_planning_trace",
        "method": "CBEP",
        "base_plan": list(final_plan),
        "article_plan": [],
        "final_plan": list(final_plan),
        "task_compatibility_assessment": {
            "incompatible_methods": list(incompatible or []),
            "incompatibility_reasons": dict(reasons or {}),
        },
    }


def _generate(
    tmp_path,
    monkeypatch,
    results,
    audit_context,
    governance_context,
    *,
    filename="report.html",
    resource_context=None,
    runtime_context=None,
    generate_pdf=False,
    provider_name=None,
):
    output = tmp_path / filename
    monkeypatch.setattr(
        report_generator,
        "build_output_path",
        lambda *args, **kwargs: str(output),
    )
    returned = report_generator.generate_report(
        results,
        audit_context.risk_tier or "high",
        provider_name=provider_name,
        audit_context=audit_context,
        governance_context=governance_context,
        resource_context=resource_context,
        runtime_context=runtime_context,
        generate_pdf=generate_pdf,
    )
    return output, output.read_text(encoding="utf-8"), returned


def _section(html, section_id):
    return html.split(f'<section id="{section_id}">', 1)[1].split("</section>", 1)[0]


def test_report_structure_status_visibility_and_summary_cards(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
    unified_result_factory,
):
    results = {
        "explainability": unified_result_factory(),
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
        "_cbep_trace": _trace(["shap", "fairness"], incompatible=["dice"]),
    }
    output, html, returned = _generate(
        tmp_path,
        monkeypatch,
        results,
        traditional_audit_context,
        governance_context,
    )

    assert returned == str(output)
    for heading in (
        "Executive Summary",
        "Runtime and Reproducibility Information",
        "CBEP Evidence Planning Summary",
        "Evidence Coverage Matrix",
        "Evidence Findings by Layer",
        "Raw Evidence Appendix",
    ):
        assert heading in html
    assert "raw-marker" in html
    assert "status-completed" in html and "status-skipped" in html
    assert "status-not-applicable" in html
    assert "Package versions" not in html
    assert "Method Applicability and Skipped Evidence" not in html
    assert "EU AI Act Article" not in _section(html, "coverage-matrix")

    executive = _section(html, "executive-summary")
    assert "System type / Modality" in executive
    assert "Task type" in executive
    assert "Governance verdict" not in executive

    findings = _section(html, "evidence-findings")
    raw = _section(html, "raw-evidence")
    assert "Skipped for test." in findings
    assert "Not applicable for test." not in findings + raw
    assert "Raw Fairness Evidence" in raw


def test_cbep_table_explains_task_and_modality_filtering():
    anomaly = AuditContext(
        system_type="traditional_ml",
        modality="tabular",
        application_domain="critical_infrastructure",
        risk_tier="high",
        task_type="anomaly_detection",
    )
    anomaly_results = {
        "_cbep_trace": _trace(
            ["shap", "drift"],
            incompatible=["fairness", "uncertainty", "lime", "dice"],
        )
    }
    anomaly_table = build_cbep_decision_table(anomaly_results, anomaly)
    anomaly_rows = {row["method"]: row for row in anomaly_table["rows"]}
    assert anomaly_rows["MAPIE"]["selected"] == "No"
    assert "IsolationForest" in anomaly_rows["MAPIE"]["reason"]

    reason = (
        "Excluded because the current DiCE implementation supports tabular "
        "counterfactuals and is not compatible with text modality."
    )
    text_context = AuditContext(
        system_type="traditional_ml",
        modality="text",
        risk_tier="high",
        task_type="binary_classification",
    )
    text_results = {
        "_cbep_trace": _trace(
            ["shap", "lime", "fairness", "uncertainty", "drift"],
            incompatible=["dice"],
            reasons={"dice": reason},
        )
    }
    text_table = build_cbep_decision_table(text_results, text_context)
    text_rows = {row["method"]: row for row in text_table["rows"]}
    assert text_rows["DiCE"]["reason"] == reason
    assert "task and modality compatibility screening" in text_table["intro"]


def test_traditional_resource_sections_are_modality_aware(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
    unified_result_factory,
):
    traditional_audit_context.model_artifact_uri = "file://data/model.joblib"
    traditional_audit_context.training_dataset_uri = "file://data/train.csv"
    traditional_audit_context.evaluation_dataset_uri = "file://data/eval.csv"
    results = {
        "explainability": unified_result_factory(),
        "_cbep_trace": _trace(["shap"]),
    }
    _, tabular_html, _ = _generate(
        tmp_path,
        monkeypatch,
        results,
        traditional_audit_context,
        governance_context,
        filename="tabular.html",
        resource_context={
            "resolved_path": r"data\case\model.joblib",
            "training_dataset_loaded": True,
            "evaluation_dataset_loaded": True,
            "training_dataset_rows": 100,
            "evaluation_dataset_rows": 25,
        },
    )
    scope = _section(tabular_html, "audit-scope")
    resources = _section(tabular_html, "resource-loading")
    runtime = _section(tabular_html, "runtime-reproducibility")
    assert scope.count("Target column") == 1
    assert "Task type" not in scope and "Model feature columns" not in scope
    assert "data/case/model.joblib" in resources
    assert "Training rows" in resources
    assert "TF-IDF vectorizer" not in resources
    assert "Selected methods" not in runtime
    assert "Model and Traditional ML dataset contract" not in runtime

    traditional_audit_context.modality = "text"
    traditional_audit_context.feature_columns = ["cv_text"]
    _, text_html, _ = _generate(
        tmp_path,
        monkeypatch,
        results,
        traditional_audit_context,
        governance_context,
        filename="text.html",
        resource_context={
            "input_adapter": "sklearn_text",
            "text_feature_column": "cv_text",
            "vectorizer_class": "TfidfVectorizer",
            "estimator_class": "LogisticRegression",
            "vocabulary_size": 742,
        },
    )
    assert "Model feature columns" in _section(text_html, "audit-scope")
    assert "TF-IDF vocabulary size" in _section(text_html, "resource-loading")

    traditional_audit_context.feature_columns = None
    _, legacy_html, _ = _generate(
        tmp_path,
        monkeypatch,
        results,
        traditional_audit_context,
        governance_context,
        filename="legacy.html",
    )
    assert "Legacy target-drop fallback" in legacy_html


def test_talentsift_report_uses_text_drift_and_ngram_semantics(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
    unified_result_factory,
):
    traditional_audit_context.modality = "text"
    traditional_audit_context.feature_columns = ["cv_text"]
    shap = unified_result_factory(
        metrics={
            "feature_semantics": "tokens_and_ngrams",
            "top_features": ["with years"],
            "feature_importance": [{"feature": "with years", "importance": 0.041466}],
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
    _, html, _ = _generate(
        tmp_path,
        monkeypatch,
        {"explainability": shap, "drift": drift, "_cbep_trace": _trace(["shap", "drift"])},
        traditional_audit_context,
        governance_context,
        provider_name="TalentSift GmbH",
    )

    expected = (
        "Canonical text-input drift was detected for the authoritative cv_text "
        "model input: three of four derived statistical characteristics were "
        "flagged as drifted, producing a drift share of 0.75."
    )
    assert expected in html
    assert "tokens_and_ngrams" in html
    assert "<th>Token / n-gram</th>" in html


def test_llm_reports_show_canonical_methods_and_resource_states(
    tmp_path,
    monkeypatch,
    governance_context,
):
    audit = AuditContext(
        system_type="agentic",
        modality="text",
        application_domain="justice",
        risk_tier="high",
        task_type="llm_generation",
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
    _, metadata_html, _ = _generate(
        tmp_path,
        monkeypatch,
        results,
        audit,
        governance_context,
        filename="metadata.html",
        resource_context={
            "model_status": "metadata_only",
            "model_is_loadable": False,
            "golden_set": [{"prompt": "test"}],
            "golden_set_loaded": True,
        },
    )
    for token in LLM_METHOD_ORDER:
        assert report_method_title(token) in metadata_html
    assert "status-metadata-only" in metadata_html
    assert "Golden-set records" in metadata_html
    assert metadata_html.count("status-skipped") >= 4

    local = AuditContext(
        system_type="llm",
        modality="text",
        application_domain="justice",
        task_type="llm_generation",
        risk_tier="high",
        model_type="distilgpt2_local_surrogate",
    )
    _, local_html, _ = _generate(
        tmp_path,
        monkeypatch,
        {"_cbep_trace": _trace([])},
        local,
        governance_context,
        filename="local.html",
        resource_context={
            "model_status": "loaded",
            "model_is_loadable": True,
            "is_local_surrogate": True,
            "runtime_mode": "local",
            "evaluation_embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        },
    )
    assert "Local surrogate execution validation" in local_html
    assert "DistilGPT2" in local_html
    assert "Evaluation embedding model" in local_html
    assert "Validation context origin" not in local_html
    assert "Governance scenario" not in local_html


def test_pdf_export_flag_controls_exporter(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
    unified_result_factory,
):
    results = {
        "explainability": unified_result_factory(),
        "_cbep_trace": _trace(["shap"]),
    }

    def reject_export(*args, **kwargs):
        raise AssertionError("PDF exporter should not be called")

    monkeypatch.setattr(report_generator, "export_html_to_pdf", reject_export)
    _generate(
        tmp_path,
        monkeypatch,
        results,
        traditional_audit_context,
        governance_context,
        filename="html_only.html",
        generate_pdf=False,
    )

    calls = []

    def fake_export(html_path, pdf_path):
        calls.append((Path(html_path), Path(pdf_path)))
        Path(pdf_path).write_bytes(b"%PDF-test")
        return Path(pdf_path).resolve()

    monkeypatch.setattr(report_generator, "export_html_to_pdf", fake_export)
    output, _, _ = _generate(
        tmp_path,
        monkeypatch,
        results,
        traditional_audit_context,
        governance_context,
        filename="with_pdf.html",
        generate_pdf=True,
    )
    assert calls == [(output, output.with_suffix(".pdf"))]
    assert output.with_suffix(".pdf").read_bytes() == b"%PDF-test"


def test_dice_report_renders_policy_all_results_and_label_meanings(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
):
    result = completed_evidence(
        evidence_id="EXP-DICE",
        layer="explainability",
        method="DiCE",
        article_mapping=["Art. 13"],
        summary="Structured counterfactual explanation.",
        metrics={
            "counterfactual_policy_source": "audit_context_immutable_exclusions",
            "counterfactual_policy_status": "validated",
            "immutable_feature_columns": ["age"],
            "features_to_vary": ["income", "duration"],
            "original_prediction": 1,
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
                    "changed_features": [{"feature": "income"}, {"feature": "duration"}],
                },
            ],
        },
        artifacts=["outputs/dice/test_counterfactuals.json"],
        limitations=["Domain review is required."],
    )
    results = {"counterfactual": result, "_cbep_trace": _trace(["dice"])}
    _, html, _ = _generate(
        tmp_path,
        monkeypatch,
        results,
        traditional_audit_context,
        governance_context,
        filename="dice.html",
    )
    assert "Explainability Evidence — DiCE Counterfactual Explanation" in html
    assert "1 (label meaning unavailable)" in html
    assert "0 (label meaning unavailable)" in html
    assert "The table below shows the first generated counterfactual" in html
    assert "All generated counterfactuals" in html
    assert "income, duration" in html
    assert "Open full structured DiCE JSON artifact" in html

    traditional_audit_context.prediction_label_mapping = {
        0: "good credit risk",
        1: "bad credit risk",
    }
    _, mapped_html, _ = _generate(
        tmp_path,
        monkeypatch,
        results,
        traditional_audit_context,
        governance_context,
        filename="dice_labels.html",
    )
    assert "1 (bad credit risk)" in mapped_html
    assert "0 (good credit risk)" in mapped_html


def test_reproducibility_hashes_are_auditable_without_internal_git_metadata(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
):
    from report import runtime_metadata
    from report.runtime_metadata import sha256_file

    paths = {}
    for name in ("s4", "s5", "model", "train", "evaluation"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("utf-8"))
        paths[name] = path
    governance_context.source_json_path = paths["s4"]
    traditional_audit_context.source_json_path = paths["s5"]
    traditional_audit_context.model_artifact_uri = str(paths["model"])
    traditional_audit_context.training_dataset_uri = str(paths["train"])
    traditional_audit_context.evaluation_dataset_uri = str(paths["evaluation"])
    monkeypatch.setattr(
        runtime_metadata,
        "collect_git_metadata",
        lambda root: {
            "status": "available",
            "commit_sha": "deadbeef" * 5,
            "branch": "private-research-branch",
            "dirty": True,
        },
    )
    _, html, _ = _generate(
        tmp_path,
        monkeypatch,
        {"_cbep_trace": _trace([])},
        traditional_audit_context,
        governance_context,
        runtime_context={
            "run_timestamp": "2026-08-06T12:30:00+02:00",
            "run_id": "fixed-run-id",
        },
    )
    full_hash = sha256_file(paths["model"])
    runtime = _section(html, "runtime-reproducibility")
    raw = _section(html, "raw-evidence")
    assert f"{full_hash[:12]}…" in runtime
    assert full_hash not in runtime and full_hash in raw
    assert "Raw Reproducibility metadata JSON" in raw
    assert "fixed-run-id" in html
    for prohibited in ("Git commit", "private-research-branch", '"dirty": true'):
        assert prohibited not in html


def test_harbour_report_distinguishes_model_and_contextual_features(
    tmp_path,
    monkeypatch,
    traditional_audit_context,
    governance_context,
):
    model_features = [f"model_feature_{index}" for index in range(13)]
    drift = completed_evidence(
        evidence_id="DRIFT-EVIDENTLY",
        layer="drift",
        method="Evidently + Feature Drift Tests",
        article_mapping=["Art. 15", "Art. 61"],
        summary="Canonical model-input drift used 13 fitted model features.",
        metrics={
            "feature_scope_source": "model_artifact_feature_cols",
            "model_feature_columns": model_features,
            "contextual_columns": ["crane_id", "visibility_m"],
            "contextual_dataset_drift": {"status": "supplementary_not_analyzed"},
            "dataset_drift_detected": False,
            "drift_share": 0.076923,
            "drifted_feature_count": 1,
            "features_analyzed": 13,
            "canonical_feature_tests": [],
        },
        raw_output={"model_feature_columns": model_features},
    )
    _, html, _ = _generate(
        tmp_path,
        monkeypatch,
        {"drift": drift, "_cbep_trace": _trace(["drift"])},
        traditional_audit_context,
        governance_context,
        provider_name="HarbourLogistik GmbH",
    )
    assert "model_artifact_feature_cols" in html
    assert "Supplementary contextual / excluded columns" in html
    assert "crane_id, visibility_m" in html
    assert "Model-input drift not detected; share 0.076923" in html
