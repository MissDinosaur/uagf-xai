from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from joblib import hash as joblib_hash
from scipy.sparse import csr_matrix, issparse
from sklearn.base import is_classifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)

from layers.uncertainty import mapie_runner
from layers.uncertainty.mapie_runner import _named_estimator_view, run_uncertainty


def test_mapie_classification_uses_prefit_estimator_and_disjoint_subsets():
    X = pd.DataFrame({"feature": np.arange(40, dtype=float)})
    y = pd.Series(np.tile([0, 1], 20))
    model = RandomForestClassifier(n_estimators=10, random_state=7).fit(X, y)
    original_estimators = len(model.estimators_)

    result = run_uncertainty(
        model,
        X,
        y,
        task_type="binary_classification",
    )

    assert result["status"] == "completed"
    assert result["prediction_interval_kind"] == "classification_prediction_set"
    assert result["estimator_mode"] == "prefit"
    assert result["calibration_rows"] == 20
    assert result["measurement_rows"] == 20
    assert len(model.estimators_) == original_estimators


def test_mapie_forecasting_returns_numeric_prediction_intervals_without_refit():
    X = pd.DataFrame({"time": np.arange(60, dtype=float)})
    y = pd.Series(0.5 * X["time"] + np.sin(X["time"] / 3.0))
    model = RandomForestRegressor(n_estimators=12, random_state=11).fit(X, y)
    original_estimators = len(model.estimators_)

    result = run_uncertainty(model, X, y, task_type="forecasting")

    assert result["status"] == "completed"
    assert result["prediction_interval_kind"] == "numeric_prediction_interval"
    assert result["task_type"] == "forecasting"
    assert 0.0 <= result["coverage"] <= 1.0
    assert result["mean_interval_width"] >= 0.0
    assert result["calibration_rows"] == 30
    assert result["measurement_rows"] == 30
    assert len(model.estimators_) == original_estimators


def test_mapie_named_classifier_delegates_without_mutating_original_estimator():
    X = pd.DataFrame({"a": np.arange(40), "b": np.arange(40) % 3})
    y = pd.Series(np.tile([0, 1], 20))
    model = GradientBoostingClassifier(random_state=5).fit(X, y)
    original_hash = joblib_hash(model)

    with (
        patch.object(model, "predict", wraps=model.predict) as predict_spy,
        patch.object(model, "predict_proba", wraps=model.predict_proba) as proba_spy,
        patch.object(model, "fit", side_effect=AssertionError("fit must not run")) as fit_spy,
    ):
        result = run_uncertainty(model, X, y, task_type="binary_classification")

    assert result["status"] == "completed"
    assert result["calibration_rows"] == 20
    assert result["measurement_rows"] == 20
    assert predict_spy.called
    assert proba_spy.called
    assert not fit_spy.called
    assert joblib_hash(model) == original_hash


def test_mapie_named_regressor_calls_original_predict_without_refit():
    X = pd.DataFrame({"a": np.arange(40), "b": np.arange(40) % 5})
    y = pd.Series(0.3 * X["a"] + X["b"])
    model = RandomForestRegressor(n_estimators=8, random_state=3).fit(X, y)
    original_hash = joblib_hash(model)

    with (
        patch.object(model, "predict", wraps=model.predict) as predict_spy,
        patch.object(model, "fit", side_effect=AssertionError("fit must not run")) as fit_spy,
    ):
        result = run_uncertainty(model, X, y, task_type="regression")

    assert result["status"] == "completed"
    assert predict_spy.called
    assert not fit_spy.called
    assert joblib_hash(model) == original_hash


def test_mapie_production_path_contains_no_clone_or_manual_inference():
    source = Path("layers/uncertainty/mapie_runner.py").read_text(encoding="utf-8")

    assert "clone(" not in source
    assert "fit_transform(" not in source
    assert "partial_fit(" not in source
    assert "coef_" not in source


def test_missing_task_type_uses_binary_classification_view(monkeypatch):
    X = pd.DataFrame({"feature": np.arange(40, dtype=float)})
    y = pd.Series(np.tile([0, 1], 20))
    model = RandomForestClassifier(n_estimators=8, random_state=13).fit(X, y)
    original_factory = mapie_runner._named_estimator_view
    captured = {}

    def capture_view(estimator, values, task_type):
        view = original_factory(estimator, values, task_type)
        captured.update(task_type=task_type, view=view)
        return view

    monkeypatch.setattr(mapie_runner, "_named_estimator_view", capture_view)
    result = run_uncertainty(model, X, y, task_type=None)

    assert captured["task_type"] == "binary_classification"
    assert is_classifier(captured["view"])
    assert result["task_type"] == "binary_classification"
    assert result["prediction_interval_kind"] == "classification_prediction_set"
    assert result["calibration_policy"] == "deterministic_stratified_half_split"


def test_named_classifier_exposes_only_supported_optional_capabilities():
    X = pd.DataFrame({"feature": np.arange(20, dtype=float)})
    y = pd.Series(np.tile([0, 1], 10))
    random_forest = RandomForestClassifier(n_estimators=5, random_state=2).fit(X, y)
    gradient_boosting = GradientBoostingClassifier(random_state=2).fit(X, y)

    forest_view = _named_estimator_view(random_forest, X, "binary_classification")
    boosting_view = _named_estimator_view(
        gradient_boosting,
        X,
        "binary_classification",
    )

    assert hasattr(forest_view, "predict_proba")
    assert not hasattr(forest_view, "decision_function")
    assert hasattr(boosting_view, "decision_function")


def test_non_text_sparse_input_is_not_silently_converted(monkeypatch):
    X = csr_matrix(np.arange(40, dtype=float).reshape(20, 2))
    y = np.tile([0, 1], 10)
    captured = {}

    def fake_uncertainty(estimator, values, labels, confidence_level, task_type):
        captured["X"] = values
        return {
            "prediction_interval_kind": "classification_prediction_set",
            "coverage": 0.9,
            "mean_interval_width": 1.0,
            "calibration_rows": 10,
            "measurement_rows": 10,
        }

    monkeypatch.setattr(mapie_runner, "_classification_uncertainty", fake_uncertainty)
    result = run_uncertainty(
        RandomForestClassifier(n_estimators=3, random_state=4).fit(X, y),
        X,
        y,
        task_type="binary_classification",
        modality="tabular",
    )

    assert issparse(captured["X"])
    assert result["input_representation"] == "model_ready_sparse"
