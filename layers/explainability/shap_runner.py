"""
Explainability module using SHAP.

This runner keeps the raw feature matrix separate from the model-ready
feature matrix:

* X_raw   -> human-readable reporting, sensitive-feature logic
* X_model -> numeric input for SHAP / model-native explainers
"""

from __future__ import annotations

import os

import numpy as np

from output_naming import build_output_path


def _is_numeric_frame(frame) -> bool:
    if not hasattr(frame, "dtypes"):
        return False

    try:
        return all(np.issubdtype(dtype, np.number) for dtype in frame.dtypes)
    except TypeError:
        return False


def _prepare_model_input(model, X):
    if hasattr(model, "prepare_input"):
        return model.prepare_input(X)
    return X


def _resolve_estimator(model):
    return getattr(model, "estimator", model)


def _get_shap_values_array(shap_values):
    values = getattr(shap_values, "values", shap_values)
    if isinstance(values, list):
        values = np.stack([np.asarray(item) for item in values], axis=-1)
    return np.asarray(values)


def _build_feature_importance(X_model, shap_values):
    values = _get_shap_values_array(shap_values)
    abs_values = np.abs(values)

    if abs_values.ndim == 3:
        importance = abs_values.mean(axis=(0, 2))
    elif abs_values.ndim == 2:
        importance = abs_values.mean(axis=0)
    elif abs_values.ndim == 1:
        importance = abs_values
    else:
        raise ValueError(
            f"Unsupported SHAP values shape: {abs_values.shape!r}"
        )

    if hasattr(X_model, "columns"):
        feature_names = list(X_model.columns)
    else:
        feature_names = [f"feature_{i}" for i in range(len(importance))]

    ranked = sorted(
        (
            {
                "feature": name,
                "importance": round(float(score), 6),
            }
            for name, score in zip(feature_names, importance)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranked


def run_shap(
    model,
    X_raw,
    X_model=None,
    estimator=None,
    provider_name=None,
    output_namespace="audit",
):
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {
            "type": "explainability",
            "method": "SHAP",
            "status": "skipped",
            "reason": (
                "SHAP and matplotlib are required to run explainability "
                f"analysis: {exc}"
            ),
        }

    os.makedirs("outputs/shap", exist_ok=True)
    output_path = build_output_path(
        "shap",
        provider_name,
        "shap_summary",
        ".png",
        fallback=output_namespace,
    )

    model_for_shap = estimator or _resolve_estimator(model)
    model_input = X_model if X_model is not None else _prepare_model_input(
        model,
        X_raw,
    )

    if not _is_numeric_frame(model_input):
        return {
            "type": "explainability",
            "method": "SHAP",
            "status": "skipped",
            "reason": (
                "Model-ready input for SHAP is not fully numeric. "
                "This usually means the artifact does not expose a compatible "
                "prepare_input() transformation."
            ),
        }

    try:
        try:
            explainer = shap.TreeExplainer(model_for_shap)
            shap_values = explainer(model_input)
            explainer_name = "TreeExplainer"
        except Exception:
            explainer = shap.Explainer(model_for_shap, model_input)
            shap_values = explainer(model_input)
            explainer_name = type(explainer).__name__

        plt.figure()
        shap.summary_plot(shap_values, model_input, show=False)
        plt.savefig(output_path)
        plt.close()

        feature_importance = _build_feature_importance(
            model_input,
            shap_values,
        )
        top_features = [item["feature"] for item in feature_importance[:10]]

        return {
            "type": "explainability",
            "method": "SHAP",
            "status": "completed",
            "explainer": explainer_name,
            "top_features": top_features,
            "feature_importance": feature_importance,
            "plot": output_path,
        }
    except Exception as exc:
        return {
            "type": "explainability",
            "method": "SHAP",
            "status": "skipped",
            "reason": (
                "SHAP execution failed on the prepared model input. "
                f"{type(exc).__name__}: {exc}"
            ),
        }

