import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from layers.uncertainty.mapie_runner import run_uncertainty


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
