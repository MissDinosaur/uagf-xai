"""Conformal uncertainty evidence for classification and numeric prediction."""

from __future__ import annotations


_CLASSIFICATION_TASKS = {
    "binary_classification",
    "multiclass_classification",
}
_REGRESSION_TASKS = {"regression", "forecasting"}


def _model_view(model, X):
    """Return the exact fitted estimator and its numeric model-ready input."""
    if hasattr(model, "prepare_input"):
        return getattr(model, "estimator", model), model.prepare_input(X)
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

    split_index = sample_count // 2
    if hasattr(X, "iloc"):
        X_calibration = X.iloc[:split_index]
        X_measurement = X.iloc[split_index:]
    else:
        X_array = np.asarray(X)
        X_calibration = X_array[:split_index]
        X_measurement = X_array[split_index:]

    if hasattr(y, "iloc"):
        y_calibration = y.iloc[:split_index]
        y_measurement = y.iloc[split_index:]
    else:
        y_array = np.asarray(y)
        y_calibration = y_array[:split_index]
        y_measurement = y_array[split_index:]

    if task_type in _CLASSIFICATION_TASKS:
        calibration_classes = set(np.asarray(y_calibration).tolist())
        observed_classes = set(np.asarray(y).tolist())
        if calibration_classes != observed_classes:
            raise ValueError(
                "The deterministic MAPIE calibration subset does not contain "
                "every observed class. Provide a larger or suitably ordered "
                "evaluation dataset."
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


def run_uncertainty(model, X, y, task_type=None):
    """Generate MAPIE evidence without refitting the S5-provided estimator."""
    if y is None:
        raise ValueError("MAPIE uncertainty analysis requires evaluation labels.")

    normalized_task = str(task_type or "").strip().lower()
    estimator, X_model = _model_view(model, X)
    confidence_level = 0.9

    try:
        if normalized_task in _REGRESSION_TASKS:
            metrics = _regression_uncertainty(
                estimator,
                X_model,
                y,
                confidence_level,
                normalized_task,
            )
        elif normalized_task in _CLASSIFICATION_TASKS or not normalized_task:
            metrics = _classification_uncertainty(
                estimator,
                X_model,
                y,
                confidence_level,
                normalized_task or "binary_classification",
            )
        else:
            raise ValueError(
                f"MAPIE does not support task type {normalized_task!r}."
            )
    except ImportError as exc:
        raise ImportError("MAPIE is required to run uncertainty analysis.") from exc
    except Exception as exc:
        raise RuntimeError(
            f"MAPIE execution failed for task_type={normalized_task!r}: {exc}"
        ) from exc

    coverage = metrics["coverage"]
    result = {
        "type": "uncertainty",
        "method": "MAPIE (Conformal Prediction)",
        "status": "completed",
        "task_type": normalized_task,
        "estimator_mode": "prefit",
        "calibration_policy": "first_half_of_evaluation_data",
        "confidence_level": confidence_level,
        **metrics,
        "coverage": round(coverage, 4),
        "mean_interval_width": round(metrics["mean_interval_width"], 4),
        "coverage_gap": round(abs(confidence_level - coverage), 4),
    }
    print("Uncertainty estimation completed.")
    return result
