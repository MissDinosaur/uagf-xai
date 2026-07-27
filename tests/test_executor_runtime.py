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
