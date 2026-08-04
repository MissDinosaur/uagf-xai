from __future__ import annotations

from pipeline import executor
from schema.evidence_schema import completed_evidence


def test_executor_records_runtime_in_existing_metrics(monkeypatch):
    evidence = completed_evidence(
        evidence_id="EXP-SHAP",
        layer="explainability",
        method="SHAP",
        article_mapping=["Art. 13"],
        summary="Test evidence.",
        key_findings=[],
        metrics={"score": 0.5},
        artifacts=[],
        limitations=[],
        raw_output={},
    )
    monkeypatch.setattr(executor, "run_shap", lambda *args, **kwargs: evidence)

    results = executor.execute(
        model=object(),
        X=[[1.0], [2.0]],
        y=[0, 1],
        methods=["shap"],
        task_type="binary_classification",
    )

    runtime = results["explainability"]["metrics"]["runtime_seconds"]
    assert isinstance(runtime, float)
    assert runtime >= 0.0


def test_fairness_is_skipped_when_s5_sensitive_features_are_missing():
    results = executor.execute(
        model=object(),
        X=[[1.0], [2.0]],
        y=[0, 1],
        methods=["fairness"],
        task_type="binary_classification",
        sensitive_features=[],
    )

    assert results["fairness"]["status"] == "skipped"
    assert "S5 did not provide sensitive_feature_columns" in results["fairness"]["summary"]
