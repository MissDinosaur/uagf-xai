"""Conformal uncertainty evidence for classification and numeric prediction."""

from __future__ import annotations


_CLASSIFICATION_TASKS = {
    "binary_classification",
    "multiclass_classification",
}
_REGRESSION_TASKS = {"regression", "forecasting"}
_MAX_TEXT_MAPIE_DENSE_BYTES = 512 * 1024 * 1024


def _named_estimator_view(estimator, X, task_type):
    """Delegate MAPIE ndarray calls to the original estimator with named columns."""
    if not hasattr(X, "columns") or not hasattr(estimator, "feature_names_in_"):
        return estimator

    import pandas as pd
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.utils.metaestimators import available_if

    feature_names = list(estimator.feature_names_in_)

    class NamedClassifierView(ClassifierMixin, BaseEstimator):
        def __init__(self, fitted_estimator, columns):
            self.fitted_estimator = fitted_estimator
            self.columns = columns

        @property
        def classes_(self):
            return self.fitted_estimator.classes_

        @property
        def n_features_in_(self):
            return len(self.columns)

        def _frame(self, values):
            if isinstance(values, pd.DataFrame):
                return values.loc[:, self.columns]
            return pd.DataFrame(values, columns=self.columns)

        def predict(self, values):
            return self.fitted_estimator.predict(self._frame(values))

        @available_if(lambda self: hasattr(self.fitted_estimator, "predict_proba"))
        def predict_proba(self, values):
            return self.fitted_estimator.predict_proba(self._frame(values))

        @available_if(lambda self: hasattr(self.fitted_estimator, "decision_function"))
        def decision_function(self, values):
            return self.fitted_estimator.decision_function(self._frame(values))

        def fit(self, X, y=None):
            raise RuntimeError("The MAPIE feature-name view is read-only.")

        def __sklearn_is_fitted__(self):
            return True

    class NamedRegressorView(RegressorMixin, BaseEstimator):
        def __init__(self, fitted_estimator, columns):
            self.fitted_estimator = fitted_estimator
            self.columns = columns

        @property
        def n_features_in_(self):
            return len(self.columns)

        def _frame(self, values):
            if isinstance(values, pd.DataFrame):
                return values.loc[:, self.columns]
            return pd.DataFrame(values, columns=self.columns)

        def predict(self, values):
            return self.fitted_estimator.predict(self._frame(values))

        def fit(self, X, y=None):
            raise RuntimeError("The MAPIE feature-name view is read-only.")

        def __sklearn_is_fitted__(self):
            return True

    if task_type in _CLASSIFICATION_TASKS:
        return NamedClassifierView(estimator, feature_names)
    return NamedRegressorView(estimator, feature_names)


def _model_view(model, X):
    """Return the exact fitted estimator and its numeric model-ready input."""
    if hasattr(model, "prepare_input"):
        estimator = getattr(model, "estimator", model)
        return estimator, model.prepare_input(X)
    return model, X


def _split_calibration_and_measurement(X, y, task_type):
    """Create disjoint deterministic conformalization and measurement subsets."""
    import numpy as np

    sample_count = len(y)
    if sample_count < 4:
        raise ValueError(
            "MAPIE requires at least four evaluation rows so calibration and "
            "measurement use disjoint subsets."
        )

    if task_type in _CLASSIFICATION_TASKS:
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.5,
            random_state=42,
        )
        calibration_indices, measurement_indices = next(
            splitter.split(np.zeros(sample_count), np.asarray(y))
        )
    else:
        split_index = sample_count // 2
        calibration_indices = np.arange(split_index)
        measurement_indices = np.arange(split_index, sample_count)

    if hasattr(X, "iloc"):
        X_calibration = X.iloc[calibration_indices]
        X_measurement = X.iloc[measurement_indices]
    else:
        X_calibration = X[calibration_indices]
        X_measurement = X[measurement_indices]
    if hasattr(y, "iloc"):
        y_calibration = y.iloc[calibration_indices]
        y_measurement = y.iloc[measurement_indices]
    else:
        y_array = np.asarray(y)
        y_calibration = y_array[calibration_indices]
        y_measurement = y_array[measurement_indices]

    if task_type in _CLASSIFICATION_TASKS:
        calibration_classes = set(np.asarray(y_calibration).tolist())
        observed_classes = set(np.asarray(y).tolist())
        if calibration_classes != observed_classes:
            raise ValueError(
                "The deterministic MAPIE calibration subset does not contain "
                "every observed class. Provide a larger or suitably ordered "
                "evaluation dataset."
            )
        measurement_classes = set(np.asarray(y_measurement).tolist())
        if measurement_classes != observed_classes:
            raise ValueError(
                "The deterministic MAPIE measurement subset does not contain "
                "every observed class."
            )

    return X_calibration, y_calibration, X_measurement, y_measurement


def _classification_uncertainty(estimator, X, y, confidence_level, task_type):
    import numpy as np
    from mapie.classification import SplitConformalClassifier

    X_cal, y_cal, X_eval, y_eval = _split_calibration_and_measurement(
        X,
        y,
        task_type,
    )
    conformal_model = SplitConformalClassifier(
        estimator=estimator,
        confidence_level=confidence_level,
        prefit=True,
    )
    conformal_model.conformalize(X_cal, y_cal)
    _, prediction_sets = conformal_model.predict_set(X_eval)

    sets = np.asarray(prediction_sets)
    if sets.ndim == 3:
        sets = sets[:, :, 0]
    classes = np.asarray(getattr(estimator, "classes_", []))
    if not len(classes):
        raise ValueError("The fitted classifier does not expose classes_.")
    class_to_index = {value: index for index, value in enumerate(classes)}
    y_values = np.asarray(y_eval)
    coverage_values = [
        bool(sets[index, class_to_index[value]])
        for index, value in enumerate(y_values)
        if value in class_to_index
    ]
    if len(coverage_values) != len(y_values):
        raise ValueError(
            "Evaluation labels contain values not present in the fitted model classes_."
        )

    return {
        "prediction_interval_kind": "classification_prediction_set",
        "coverage": float(np.mean(coverage_values)),
        "mean_interval_width": float(np.mean(sets.sum(axis=1))),
        "calibration_rows": len(y_cal),
        "measurement_rows": len(y_eval),
    }


def _regression_uncertainty(estimator, X, y, confidence_level, task_type):
    import numpy as np
    from mapie.regression import SplitConformalRegressor

    X_cal, y_cal, X_eval, y_eval = _split_calibration_and_measurement(
        X,
        y,
        task_type,
    )
    conformal_model = SplitConformalRegressor(
        estimator=estimator,
        confidence_level=confidence_level,
        prefit=True,
    )
    conformal_model.conformalize(X_cal, y_cal)
    _, intervals = conformal_model.predict_interval(X_eval)

    interval_array = np.asarray(intervals, dtype=float)
    if interval_array.ndim == 3:
        interval_array = interval_array[:, :, 0]
    if interval_array.ndim != 2 or interval_array.shape[1] != 2:
        raise ValueError(
            f"Unexpected MAPIE regression interval shape: {interval_array.shape}."
        )
    lower = interval_array[:, 0]
    upper = interval_array[:, 1]
    y_values = np.asarray(y_eval, dtype=float)

    return {
        "prediction_interval_kind": "numeric_prediction_interval",
        "coverage": float(np.mean((y_values >= lower) & (y_values <= upper))),
        "mean_interval_width": float(np.mean(upper - lower)),
        "calibration_rows": len(y_cal),
        "measurement_rows": len(y_eval),
        "interval_min": float(np.min(lower)),
        "interval_max": float(np.max(upper)),
    }


def run_uncertainty(model, X, y, task_type=None, modality=None):
    """Generate MAPIE evidence without refitting the S5-provided estimator."""
    if y is None:
        raise ValueError("MAPIE uncertainty analysis requires evaluation labels.")

    normalized_task = str(task_type or "").strip().lower()
    effective_task = normalized_task or "binary_classification"
    estimator, X_model = _model_view(model, X)
    if hasattr(X_model, "columns") and hasattr(estimator, "feature_names_in_"):
        feature_names = list(estimator.feature_names_in_)
        missing = [name for name in feature_names if name not in X_model.columns]
        if missing:
            raise ValueError(f"MAPIE input is missing fitted model features: {missing}")
        X_model = X_model.loc[:, feature_names]
    mapie_estimator = _named_estimator_view(estimator, X_model, effective_task)
    is_text = str(modality or "").strip().lower() == "text"
    input_representation = "tfidf_dense" if is_text else "model_ready_numeric"
    try:
        from scipy.sparse import issparse

        if issparse(X_model):
            if is_text:
                # MAPIE 1.3.0 calls len(X) and rejects SciPy sparse text matrices.
                dense_bytes = int(
                    X_model.shape[0] * X_model.shape[1] * X_model.dtype.itemsize
                )
                if dense_bytes > _MAX_TEXT_MAPIE_DENSE_BYTES:
                    dense_mib = dense_bytes / (1024 * 1024)
                    limit_mib = _MAX_TEXT_MAPIE_DENSE_BYTES / (1024 * 1024)
                    raise MemoryError(
                        f"MAPIE requires dense text input in the installed API, but "
                        f"the projected TF-IDF matrix is {dense_mib:.1f} MiB and "
                        f"exceeds the {limit_mib:.0f} MiB safety limit."
                    )
                X_model = X_model.toarray()
                input_representation = "tfidf_dense_for_mapie"
            else:
                input_representation = "model_ready_sparse"
    except ImportError:
        pass
    confidence_level = 0.9

    try:
        if effective_task in _REGRESSION_TASKS:
            metrics = _regression_uncertainty(
                mapie_estimator,
                X_model,
                y,
                confidence_level,
                effective_task,
            )
        elif effective_task in _CLASSIFICATION_TASKS:
            metrics = _classification_uncertainty(
                mapie_estimator,
                X_model,
                y,
                confidence_level,
                effective_task,
            )
        else:
            raise ValueError(f"MAPIE does not support task type {effective_task!r}.")
    except ImportError as exc:
        raise ImportError("MAPIE is required to run uncertainty analysis.") from exc
    except Exception as exc:
        raise RuntimeError(
            f"MAPIE execution failed for task_type={effective_task!r}: {exc}"
        ) from exc

    coverage = metrics["coverage"]
    result = {
        "type": "uncertainty",
        "method": "MAPIE (Conformal Prediction)",
        "status": "completed",
        "task_type": effective_task,
        "input_representation": input_representation,
        "estimator_class": type(getattr(model, "estimator", estimator)).__name__,
        "estimator_mode": "prefit",
        "calibration_policy": (
            "deterministic_stratified_half_split"
            if effective_task in _CLASSIFICATION_TASKS
            else "first_half_of_evaluation_data"
        ),
        "confidence_level": confidence_level,
        **metrics,
        "coverage": round(coverage, 4),
        "mean_interval_width": round(metrics["mean_interval_width"], 4),
        "mean_prediction_set_size": round(metrics["mean_interval_width"], 4)
        if effective_task in _CLASSIFICATION_TASKS
        else None,
        "coverage_gap": round(abs(confidence_level - coverage), 4),
    }
    print("Uncertainty estimation completed.")
    return result
