from __future__ import annotations

import pandas as pd
import pytest

from layers.drift import evidently_runner
from pipeline.evidence_normalizer import REQUIRED_FIELDS


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
