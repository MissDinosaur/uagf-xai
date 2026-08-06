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


def _run_text_shap(model, X_raw, shap, plt, output_path):
    required = ("prepare_input", "get_feature_names_out", "predict_proba")
    if any(not hasattr(model, name) for name in required):
        raise TypeError(
            "Text SHAP requires a fitted sklearn text model adapter with "
            "TF-IDF feature names and probabilities."
        )

    X_tfidf = model.prepare_input(X_raw)
    estimator = model.estimator
    feature_names = np.asarray(model.get_feature_names_out(), dtype=object)
    row_count = X_tfidf.shape[0]
    if row_count == 0:
        raise ValueError("Text SHAP requires at least one evaluation record.")

    rng = np.random.default_rng(42)
    background_count = min(50, row_count)
    background_indices = np.sort(
        rng.choice(row_count, size=background_count, replace=False)
    )
    explanation_count = min(50, row_count)
    explanation_indices = np.arange(explanation_count)
    background = X_tfidf[background_indices]
    explained = X_tfidf[explanation_indices]

    # SHAP 0.49 accepts the original fitted LogisticRegression directly, so no
    # coefficient tuple or replacement estimator is required.
    explainer = shap.LinearExplainer(estimator, background)
    shap_values = explainer(explained)
    values = _get_shap_values_array(shap_values)
    if values.ndim == 3:
        global_scores = np.abs(values).mean(axis=(0, 2))
    elif values.ndim == 2:
        global_scores = np.abs(values).mean(axis=0)
    else:
        raise ValueError(f"Unexpected text SHAP value shape: {values.shape!r}")

    ranked_indices = np.argsort(global_scores)[::-1]
    global_importance = [
        {
            "token": str(feature_names[index]),
            "importance": round(float(global_scores[index]), 6),
        }
        for index in ranked_indices[:50]
        if global_scores[index] > 0
    ]

    probabilities = np.asarray(model.predict_proba(X_raw.iloc[[0]]))[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = model.classes_[predicted_index]
    local_values = values[0]
    if local_values.ndim == 2:
        local_values = local_values[:, predicted_index]
    present_indices = set(explained[0].nonzero()[1].tolist())
    local_ranked = sorted(
        present_indices,
        key=lambda index: abs(float(local_values[index])),
        reverse=True,
    )
    local_attributions = [
        {
            "token": str(feature_names[index]),
            "shap_value": round(float(local_values[index]), 6),
            "tfidf_value": round(float(explained[0, index]), 6),
        }
        for index in local_ranked[:20]
    ]
    top_positive = sorted(
        [item for item in local_attributions if item["shap_value"] > 0],
        key=lambda item: item["shap_value"],
        reverse=True,
    )[:10]
    top_negative = sorted(
        [item for item in local_attributions if item["shap_value"] < 0],
        key=lambda item: item["shap_value"],
    )[:10]

    plot_items = global_importance[:15][::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(
        [item["token"] for item in plot_items],
        [item["importance"] for item in plot_items],
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Global TF-IDF token importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "type": "explainability",
        "method": "SHAP",
        "status": "completed",
        "explainer": "LinearExplainer",
        "input_representation": "tfidf_sparse",
        "feature_semantics": "tokens",
        "global_token_importance": global_importance,
        "local_token_attributions": local_attributions,
        "top_positive_tokens": top_positive,
        "top_negative_tokens": top_negative,
        "predicted_label": predicted_label.item()
        if hasattr(predicted_label, "item")
        else predicted_label,
        "class_probabilities": [round(float(value), 6) for value in probabilities],
        "top_features": [item["token"] for item in global_importance[:10]],
        "feature_importance": [
            {"feature": item["token"], "importance": item["importance"]}
            for item in global_importance
        ],
        "plot": output_path,
    }


def run_shap(
    model,
    X_raw,
    X_model=None,
    estimator=None,
    provider_name=None,
    output_namespace="audit",
    modality=None,
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
    suffix = "shap_text_tokens" if modality == "text" else "shap_summary"
    output_path = build_output_path(
        "shap",
        provider_name,
        suffix,
        ".png",
        fallback=output_namespace,
    )

    if str(modality or "").strip().lower() == "text":
        try:
            return _run_text_shap(model, X_raw, shap, plt, output_path)
        except Exception as exc:
            return {
                "type": "explainability",
                "method": "SHAP",
                "status": "failed",
                "error_type": type(exc).__name__,
                "reason": f"Text SHAP execution failed: {exc}",
            }

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
