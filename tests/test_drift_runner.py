from __future__ import annotations

import pandas as pd
import pytest

from adapters.s5_audit_adapter import AuditContext
from layers.drift import evidently_runner
from pipeline.evidence_normalizer import REQUIRED_FIELDS
from report.report_builder import build_evidence_coverage_matrix


REAL_RUN_EVIDENTLY = evidently_runner._run_evidently


def _snapshot(feature_results, engine_threshold=0.5):
    drifted_count = sum(item[3] for item in feature_results)
    metric_results = {
        "count": {
            "metric_value_location": {
                "metric": {
                    "params": {
                        "type": "evidently:metric_v2:DriftedColumnsCount",
                        "drift_share": engine_threshold,
                    }
                }
            },
            "count": {"value": drifted_count},
            "share": {"value": drifted_count / len(feature_results)},
        }
    }
    for index, (feature, value, threshold, _) in enumerate(feature_results):
        metric_results[f"feature-{index}"] = {
            "metric_value_location": {
                "metric": {
                    "params": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": feature,
                        "method": "K-S p_value",
                        "threshold": threshold,
                    }
                }
            },
            "value": value,
        }
    return {"metric_results": metric_results}


@pytest.fixture(autouse=True)
def isolate_external_drift_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evidently_runner,
        "_run_evidently",
        lambda reference, current: ({"engine": "mock-evidently"}, None),
    )
    monkeypatch.setattr(
        evidently_runner,
        "_save_feature_tests",
        lambda tests, provider_name, output_namespace: str(tmp_path / "drift.json"),
    )


def test_numeric_drift_is_detected():
    reference = pd.DataFrame({"value": list(range(20))})
    current = pd.DataFrame({"value": list(range(100, 120))})

    result = evidently_runner.run_drift(current, reference_data=reference)

    assert result["metrics"]["dataset_drift_detected"] is True
    assert result["metrics"]["drifted_features"] == ["value"]
    assert result["metrics"]["top_drifted_columns"][0]["method"] == "ks_test"


def test_identical_distributions_do_not_trigger_drift():
    reference = pd.DataFrame({"value": list(range(20)), "group": ["a", "b"] * 10})

    result = evidently_runner.run_drift(reference.copy(), reference_data=reference)

    assert result["metrics"]["dataset_drift_detected"] is False
    assert result["metrics"]["drift_share"] == 0.0
    assert result["metrics"]["drifted_features"] == []


def test_categorical_distribution_change_is_detected():
    reference = pd.DataFrame({"group": ["a"] * 20})
    current = pd.DataFrame({"group": ["b"] * 20})

    result = evidently_runner.run_drift(current, reference_data=reference)
    feature = result["metrics"]["top_drifted_columns"][0]

    assert feature["feature_type"] == "categorical"
    assert feature["method"] == "jensen_shannon"
    assert feature["p_value"] is None
    assert feature["drift_detected"] is True


def test_target_column_is_excluded_from_analysis():
    reference = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 0, 0, 0]})
    current = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [1, 1, 1, 1]})

    result = evidently_runner.run_drift(
        current,
        reference_data=reference,
        target_column="target",
    )

    assert result["metrics"]["features_analyzed"] == 1
    assert "target" not in result["metrics"]["drifted_features"]
    assert result["raw_output"]["excluded_columns"] == ["target"]


def test_drift_result_uses_unified_schema_and_required_metrics():
    reference = pd.DataFrame({"value": [1, 2, 3, 4]})
    result = evidently_runner.run_drift(reference.copy(), reference_data=reference)

    expected_metrics = {
        "dataset_drift_detected",
        "drift_share",
        "features_analyzed",
        "number_of_columns",
        "reference_rows",
        "current_rows",
        "drifted_features",
        "top_drifted_columns",
        "method_details",
    }
    assert REQUIRED_FIELDS <= set(result)
    assert expected_metrics <= set(result["metrics"])
    assert result["status"] == "completed"
    assert result["limitations"]


@pytest.mark.parametrize(
    ("current", "reference"),
    [(None, pd.DataFrame({"x": [1]})), (pd.DataFrame({"x": [1]}), None)],
)
def test_missing_reference_or_current_data_returns_skipped(current, reference):
    result = evidently_runner.run_drift(current, reference_data=reference)

    assert result["status"] == "skipped"
    assert result["metrics"] == {}
    assert "Reference and current datasets" in result["limitations"][0]


def test_successful_evidently_result_is_canonical_when_fallback_disagrees(monkeypatch):
    snapshot = _snapshot([("a", 0.01, 0.05, True), ("b", 0.8, 0.05, False)])
    monkeypatch.setattr(evidently_runner, "_run_evidently", lambda *args: (snapshot, None))
    reference = pd.DataFrame({"a": range(20), "b": range(20)})
    captured = {}
    monkeypatch.setattr(
        evidently_runner,
        "_save_feature_tests",
        lambda payload, *_: captured.update(payload) or "drift.json",
    )

    result = evidently_runner.run_drift(reference.copy(), reference_data=reference)
    metrics = result["metrics"]

    assert metrics["canonical_source"] == "evidently"
    assert metrics["drifted_features"] == ["a"]
    assert metrics["drifted_feature_count"] == 1
    assert metrics["drift_share"] == 0.5
    assert metrics["dataset_drift_detected"] is True
    assert metrics["uagf_dataset_drift_share_threshold"] == 0.2
    assert not any(
        item["drift_detected"]
        for item in metrics["supplementary_feature_tests"]["results"]
    )
    assert captured["canonical_source"] == "evidently"
    assert captured["drifted_feature_count"] == 1


def test_fallback_becomes_canonical_only_when_evidently_cannot_be_extracted(monkeypatch):
    monkeypatch.setattr(
        evidently_runner,
        "_run_evidently",
        lambda *args: ({"metric_results": {}}, None),
    )
    reference = pd.DataFrame({"value": range(20)})
    current = pd.DataFrame({"value": range(100, 120)})

    result = evidently_runner.run_drift(current, reference_data=reference)

    assert result["metrics"]["canonical_source"] == "fallback_feature_tests"
    assert result["metrics"]["drifted_features"] == ["value"]
    assert result["metrics"]["dataset_drift_detected"] is True
    assert any("could not provide canonical" in item for item in result["limitations"])


def test_report_drift_summary_uses_the_same_canonical_values(monkeypatch):
    snapshot = _snapshot([("shifted", 0.01, 0.05, True)])
    monkeypatch.setattr(evidently_runner, "_run_evidently", lambda *args: (snapshot, None))
    reference = pd.DataFrame({"shifted": range(20)})
    result = evidently_runner.run_drift(reference.copy(), reference_data=reference)
    rows = build_evidence_coverage_matrix(
        {
            "drift": result,
            "_cbep_trace": {"final_plan": ["drift"]},
        },
        AuditContext(system_type="traditional_ml", task_type="regression"),
    )

    assert "1 feature(s) were flagged as drifted" in result["summary"]
    assert rows[-1]["main_output"] == "Dataset drift detected; share 1.0"


@pytest.mark.parametrize(
    ("reference_path", "current_path", "target", "expected_count", "expected_share", "detected"),
    [
        (
            "data/01_finclear_gmbh/datasets/finclear_training_dataset.csv",
            "data/01_finclear_gmbh/datasets/finclear_evaluation_dataset.csv",
            "credit_risk",
            2,
            0.1,
            False,
        ),
        (
            "data/02_retailiq_ag/datasets/retailiq_training_dataset.csv",
            "data/02_retailiq_ag/datasets/retailiq_evaluation_dataset.csv",
            "sales",
            4,
            4 / 11,
            True,
        ),
    ],
)
def test_real_case_evidently_and_report_values_are_internally_consistent(
    monkeypatch,
    reference_path,
    current_path,
    target,
    expected_count,
    expected_share,
    detected,
):
    monkeypatch.setattr(evidently_runner, "_run_evidently", REAL_RUN_EVIDENTLY)
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    result = evidently_runner.run_drift(
        current,
        reference_data=reference,
        target_column=target,
    )
    metrics = result["metrics"]
    rows = build_evidence_coverage_matrix(
        {"drift": result, "_cbep_trace": {"final_plan": ["drift"]}},
        AuditContext(system_type="traditional_ml", task_type="regression"),
    )

    assert metrics["canonical_source"] == "evidently"
    assert metrics["drifted_feature_count"] == expected_count
    assert metrics["drift_share"] == pytest.approx(expected_share, abs=1e-6)
    assert metrics["dataset_drift_detected"] is detected
    assert sum(
        item["drift_detected"]
        for item in metrics["supplementary_feature_tests"]["results"]
    ) == 0
    assert str(metrics["drift_share"]) in rows[-1]["main_output"]
