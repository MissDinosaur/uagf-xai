"""Local tabular explanations using LIME.

LIME operates on the numeric, model-ready feature view. Raw input data remains
available to fairness and reporting through the pipeline's separate X_raw view.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from output_naming import build_output_path


_CLASSIFICATION_TASKS = {
    "binary_classification",
    "multiclass_classification",
}


def _numeric_frame(X_model) -> pd.DataFrame:
    if isinstance(X_model, pd.DataFrame):
        frame = X_model.copy()
    else:
        frame = pd.DataFrame(X_model)

    non_numeric = [
        column for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(
            "LIME requires numeric model-ready input. Non-numeric columns: "
            f"{non_numeric}"
        )
    if frame.empty:
        raise ValueError("LIME requires at least one evaluation row.")
    return frame


def _prediction_frame(values, feature_names):
    return pd.DataFrame(values, columns=feature_names)


def _scalar(value):
    array = np.asarray(value).reshape(-1)
    return float(array[0]) if len(array) else None


def _run_text_lime(model, X_model, provider_name, output_namespace):
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError as exc:
        raise ImportError("lime is required to run text LIME analysis.") from exc

    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise TypeError(
            "Text LIME requires a fitted text model adapter with predict_proba()."
        )
    if isinstance(X_model, pd.DataFrame):
        if X_model.shape[1] != 1:
            raise ValueError("Text LIME requires exactly one model input column.")
        raw_texts = X_model.iloc[:, 0].tolist()
    elif isinstance(X_model, pd.Series):
        raw_texts = X_model.tolist()
    else:
        raw_texts = list(X_model)
    if not raw_texts or any(not isinstance(value, str) for value in raw_texts):
        raise ValueError("Text LIME requires at least one non-null text record.")

    class_names = [str(value) for value in model.classes_]
    explainer = LimeTextExplainer(class_names=class_names, random_state=42)

    def predict_fn(texts):
        return model.predict_proba(list(texts))

    sample = raw_texts[0]
    available_terms = max(1, len(set(sample.split())))
    explanation = explainer.explain_instance(
        sample,
        predict_fn,
        num_features=min(10, available_terms),
        top_labels=1,
    )
    probabilities = np.asarray(predict_fn([sample]))[0]
    predicted_index = int(np.argmax(probabilities))
    labels = explanation.available_labels()
    explained_label = int(labels[0]) if labels else predicted_index
    contributions = explanation.as_list(label=explained_label)

    os.makedirs("outputs/lime", exist_ok=True)
    output_path = build_output_path(
        "lime",
        provider_name,
        "lime_text_explanation",
        ".html",
        fallback=output_namespace,
    )
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(explanation.as_html())

    predicted_label = model.classes_[predicted_index]
    if hasattr(predicted_label, "item"):
        predicted_label = predicted_label.item()
    token_contributions = [
        {"token_or_phrase": str(token), "weight": round(float(weight), 6)}
        for token, weight in contributions
    ]
    return {
        "type": "explainability",
        "method": "LIME",
        "status": "completed",
        "mode": "classification",
        "input_representation": "raw_text",
        "explainer": "LimeTextExplainer",
        "sample_index": 0,
        "predicted_label": predicted_label,
        "class_probabilities": [round(float(value), 6) for value in probabilities],
        "token_contributions": token_contributions,
        "feature_contributions": [
            {
                "feature_condition": item["token_or_phrase"],
                "weight": item["weight"],
            }
            for item in token_contributions
        ],
        "features_explained": len(token_contributions),
        "local_fidelity_score": round(float(explanation.score), 6),
        "local_prediction": round(_scalar(explanation.local_pred), 6),
        "prediction": {
            "predicted_label": predicted_label,
            "class_probabilities": [
                round(float(value), 6) for value in probabilities
            ],
        },
        "output": output_path,
    }


def run_lime(
    model,
    X_model,
    task_type,
    modality=None,
    provider_name=None,
    output_namespace="audit",
):
    """Generate a local LIME explanation for the first evaluation sample."""
    if str(modality or "").strip().lower() == "text":
        return _run_text_lime(
            model,
            X_model,
            provider_name,
            output_namespace,
        )

    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError(
            "lime is required to run LIME explainability analysis."
        ) from exc

    prepared_input = (
        model.prepare_input(X_model) if hasattr(model, "prepare_input") else X_model
    )
    estimator = getattr(model, "estimator", model)
    frame = _numeric_frame(prepared_input)
    feature_names = [str(column) for column in frame.columns]
    training_data = frame.to_numpy(dtype=float)
    normalized_task = str(task_type or "").strip().lower()
    mode = (
        "classification"
        if normalized_task in _CLASSIFICATION_TASKS
        else "regression"
    )

    if mode == "classification" and not hasattr(estimator, "predict_proba"):
        raise AttributeError(
            "LIME classification analysis requires predict_proba() on the "
            "underlying estimator."
        )
    if mode == "regression" and not hasattr(estimator, "predict"):
        raise AttributeError(
            "LIME regression analysis requires predict() on the underlying estimator."
        )

    class_names = None
    if mode == "classification":
        class_names = [
            str(value) for value in getattr(estimator, "classes_", [])
        ] or None

    explainer = LimeTabularExplainer(
        training_data=training_data,
        feature_names=feature_names,
        class_names=class_names,
        mode=mode,
        discretize_continuous=True,
        random_state=42,
    )

    sample = training_data[0]
    num_features = min(10, len(feature_names))

    try:
        if mode == "classification":
            def predict_fn(values):
                return estimator.predict_proba(
                    _prediction_frame(values, feature_names)
                )

            explanation = explainer.explain_instance(
                sample,
                predict_fn,
                num_features=num_features,
                top_labels=1,
            )
            labels = explanation.available_labels()
            explained_label = int(labels[0]) if labels else None
            contributions = explanation.as_list(label=explained_label)
            probabilities = np.asarray(
                predict_fn(sample.reshape(1, -1))
            )[0]
            predicted_index = int(np.argmax(probabilities))
            predicted_label = (
                class_names[predicted_index]
                if class_names and predicted_index < len(class_names)
                else predicted_index
            )
            prediction = {
                "predicted_label": predicted_label,
                "class_probabilities": [
                    round(float(value), 6) for value in probabilities
                ],
            }
        else:
            def predict_fn(values):
                return estimator.predict(
                    _prediction_frame(values, feature_names)
                )

            explanation = explainer.explain_instance(
                sample,
                predict_fn,
                num_features=num_features,
            )
            contributions = explanation.as_list()
            prediction = {
                "predicted_value": round(
                    _scalar(predict_fn(sample.reshape(1, -1))),
                    6,
                )
            }
    except Exception as exc:
        raise RuntimeError(
            "LIME execution failed on the numeric model-ready input. "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    os.makedirs("outputs/lime", exist_ok=True)
    output_path = build_output_path(
        "lime",
        provider_name,
        "lime_explanation",
        ".html",
        fallback=output_namespace,
    )
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(explanation.as_html())

    return {
        "type": "explainability",
        "method": "LIME",
        "status": "completed",
        "mode": mode,
        "sample_index": 0,
        "features_explained": len(contributions),
        "input_representation": "model_ready_numeric",
        "feature_contributions": [
            {
                "feature_condition": str(condition),
                "weight": round(float(weight), 6),
            }
            for condition, weight in contributions
        ],
        "local_fidelity_score": round(float(explanation.score), 6),
        "local_prediction": round(_scalar(explanation.local_pred), 6),
        "prediction": prediction,
        "output": output_path,
    }
