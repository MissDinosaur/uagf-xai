"""Generate structured DiCE counterfactual explanation evidence."""

from __future__ import annotations

import json
import math
import os
from numbers import Number
from typing import Any

import pandas as pd

from output_naming import build_output_path


def _encode_for_dice(X: pd.DataFrame):
    """Build a stable generation schema while retaining display labels."""
    categorical_dtypes = {}
    categorical_ranges = {}
    encoded = X.copy()
    for column in encoded.columns:
        if isinstance(encoded[column].dtype, pd.CategoricalDtype):
            categorical_dtypes[column] = encoded[column].dtype
            encoded[column] = encoded[column].cat.codes.astype(str).astype(object)
        elif encoded[column].dtype == object:
            categorical = encoded[column].astype("category")
            categorical_dtypes[column] = categorical.dtype
            encoded[column] = categorical.cat.codes.astype(str).astype(object)
        else:
            encoded[column] = pd.to_numeric(encoded[column], errors="raise").astype(float)
        if column in categorical_dtypes:
            categorical_ranges[column] = [
                str(code) for code in range(len(categorical_dtypes[column].categories))
            ]
    return encoded, categorical_dtypes, categorical_ranges


def _resolve_feature_policy(
    model_features,
    sensitive_features,
    actionable_feature_columns,
    immutable_feature_columns,
    target_column,
):
    """Validate normalized audit metadata and return the DiCE feature allowlist."""
    model_features = list(model_features)
    model_feature_set = set(model_features)
    sensitive = list(dict.fromkeys(sensitive_features or []))
    immutable = list(immutable_feature_columns or [])

    for field_name, values in (
        ("actionable_feature_columns", actionable_feature_columns),
        ("immutable_feature_columns", immutable),
    ):
        if values is None:
            continue
        if len(values) != len(set(values)):
            raise ValueError(f"{field_name} must not contain duplicate columns.")
        if target_column and target_column in values:
            raise ValueError(
                f"{field_name} must not contain target_column {target_column!r}."
            )
        unknown = [feature for feature in values if feature not in model_feature_set]
        if unknown:
            raise ValueError(f"{field_name} contains unknown model features: {unknown}")

    excluded_sensitive = [feature for feature in model_features if feature in sensitive]
    excluded_immutable = [feature for feature in model_features if feature in immutable]
    if actionable_feature_columns is not None:
        sensitive_conflicts = [
            feature
            for feature in actionable_feature_columns
            if feature in excluded_sensitive
        ]
        if sensitive_conflicts:
            raise ValueError(
                "actionable_feature_columns must not authorize sensitive features: "
                f"{sensitive_conflicts}"
            )
        immutable_conflicts = [
            feature
            for feature in actionable_feature_columns
            if feature in excluded_immutable
        ]
        if immutable_conflicts:
            raise ValueError(
                "Counterfactual features cannot be both actionable and immutable: "
                f"{immutable_conflicts}"
            )
        features_to_vary = list(actionable_feature_columns)
        source = "audit_context_actionable_allowlist"
        excluded_non_actionable = [
            feature
            for feature in model_features
            if feature not in features_to_vary
            and feature not in excluded_sensitive
            and feature not in excluded_immutable
        ]
        status = "validated"
    else:
        blocked = set(excluded_sensitive) | set(excluded_immutable)
        features_to_vary = [
            feature for feature in model_features if feature not in blocked
        ]
        excluded_non_actionable = []
        source = (
            "audit_context_immutable_exclusions"
            if immutable
            else "model_features_default"
        )
        status = "validated" if immutable else "metadata_not_provided"

    return {
        "counterfactual_policy_source": source,
        "counterfactual_policy_status": status,
        "actionable_feature_columns": (
            list(actionable_feature_columns)
            if actionable_feature_columns is not None
            else None
        ),
        "immutable_feature_columns": immutable,
        "excluded_sensitive_features": excluded_sensitive,
        "excluded_immutable_features": excluded_immutable,
        "excluded_non_actionable_features": excluded_non_actionable,
        "features_to_vary": features_to_vary,
    }


class _RecodingWrapper:
    """Let DiCE use integer codes while the audited model receives raw labels."""

    def __init__(self, model, categorical_dtypes):
        self._model = model
        self._categorical_dtypes = categorical_dtypes

    def _recode(self, X):
        output = pd.DataFrame(X).copy()
        for column, dtype in self._categorical_dtypes.items():
            if column in output.columns:
                output[column] = pd.Categorical.from_codes(
                    output[column].astype(int), dtype.categories
                )
        return output

    def predict(self, X):
        return self._model.predict(self._recode(X))

    def predict_proba(self, X):
        return self._model.predict_proba(self._recode(X))


def _json_value(value: Any):
    """Convert pandas/numpy scalar values into strict JSON-compatible values."""
    if value is pd.NA or value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _prediction(model, X: pd.DataFrame) -> tuple[Any, list[Any] | None]:
    prediction = _json_value(model.predict(X)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = [_json_value(value) for value in model.predict_proba(X)[0]]
    return prediction, probabilities


def _decode_value(column: str, value: Any, categorical_dtypes) -> tuple[Any, bool]:
    """Return a human-readable value and whether decoding was reliable."""
    dtype = categorical_dtypes.get(column)
    if dtype is None:
        return _json_value(value), True
    try:
        code = int(value)
        if float(value) != float(code) or code < 0 or code >= len(dtype.categories):
            return _json_value(value), False
        return _json_value(dtype.categories[code]), True
    except (TypeError, ValueError, OverflowError):
        return _json_value(value), False


def _values_equal(original: Any, counterfactual: Any) -> bool:
    if pd.isna(original) and pd.isna(counterfactual):
        return True
    if isinstance(original, Number) and isinstance(counterfactual, Number):
        return math.isclose(float(original), float(counterfactual), rel_tol=1e-9, abs_tol=1e-12)
    return str(original) == str(counterfactual)


def _changed_features(
    original_raw: pd.Series,
    original_model_ready: pd.Series,
    counterfactual_model_ready: pd.Series,
    categorical_dtypes,
    prohibited_features: set[str],
) -> tuple[list[dict[str, Any]], bool]:
    changes = []
    decoding_reliable = True
    for feature in original_model_ready.index:
        original_ready = original_model_ready[feature]
        counterfactual_ready = counterfactual_model_ready[feature]
        if _values_equal(original_ready, counterfactual_ready):
            continue

        original_display = original_raw.get(feature, original_ready)
        counterfactual_display, decoded = _decode_value(
            feature, counterfactual_ready, categorical_dtypes
        )
        decoding_reliable = decoding_reliable and decoded
        feature_type = "categorical" if feature in categorical_dtypes else "numeric"
        delta = None
        if feature_type == "numeric" and isinstance(original_ready, Number) and isinstance(counterfactual_ready, Number):
            delta = _json_value(float(counterfactual_ready) - float(original_ready))

        changes.append(
            {
                "feature": str(feature),
                "feature_type": feature_type,
                "original_value": _json_value(original_display),
                "counterfactual_value": counterfactual_display,
                "delta": delta,
                "model_ready_original_value": _json_value(original_ready),
                "model_ready_counterfactual_value": _json_value(counterfactual_ready),
                "is_sensitive_or_immutable": feature in prohibited_features,
            }
        )
    return changes, decoding_reliable


def _extract_counterfactual_frame(dice_exp, outcome_name: str) -> pd.DataFrame:
    """Extract the documented DiCE final counterfactual dataframe."""
    examples = getattr(dice_exp, "cf_examples_list", None) or []
    frames = []
    for example in examples:
        frame = getattr(example, "final_cfs_df", None)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frames.append(frame.copy())
    if not frames:
        raise RuntimeError(
            "DiCE returned no structured counterfactual rows in "
            "cf_examples_list[*].final_cfs_df."
        )
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop(columns=[outcome_name], errors="ignore")


def run_dice(
    model,
    X,
    y,
    provider_name=None,
    output_namespace="audit",
    sensitive_features=None,
    actionable_feature_columns=None,
    immutable_feature_columns=None,
    target_column=None,
):
    """Run DiCE and persist machine-readable counterfactual evidence."""
    try:
        import dice_ml
    except ImportError as exc:
        raise ImportError("DiCE is required to run counterfactual analysis.") from exc

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "DiCE requires a probabilistic classification model with predict_proba()."
        )
    if y is None:
        raise ValueError("DiCE requires evaluation labels for its data interface.")

    X_raw = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if X_raw.empty:
        raise ValueError("DiCE requires at least one evaluation instance.")

    os.makedirs("outputs/dice", exist_ok=True)
    output_path = build_output_path(
        "dice",
        provider_name,
        "counterfactuals",
        ".json",
        fallback=output_namespace,
    )

    X_encoded, categorical_dtypes, categorical_ranges = _encode_for_dice(X_raw)
    policy = _resolve_feature_policy(
        X_encoded.columns,
        sensitive_features,
        actionable_feature_columns,
        immutable_feature_columns,
        target_column,
    )
    permitted_features = policy["features_to_vary"]
    if not permitted_features:
        reason = (
            "DiCE was not executed because the validated counterfactual policy "
            "authorizes no model features to vary."
        )
        artifact = {
            "status": "skipped",
            "reason": reason,
            **policy,
            "policy_violation_detected": False,
            "sensitive_or_immutable_change_detected": False,
            "limitations": [reason],
        }
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(artifact, output_file, indent=2, ensure_ascii=False)
        return {
            "type": "explainability",
            "method": "DiCE",
            "evidence_type": "counterfactual_explanation",
            "output": output_path,
            **artifact,
        }

    outcome_name = "__uagf_target__"
    dataframe = X_encoded.reset_index(drop=True).copy()
    labels = pd.Series(y).reset_index(drop=True)
    if len(labels) != len(dataframe):
        raise ValueError(
            "DiCE requires X and y to contain the same number of evaluation rows."
        )
    dataframe[outcome_name] = labels

    categorical_columns = set(categorical_dtypes)
    continuous_columns = [
        column for column in X_encoded.columns if column not in categorical_columns
    ]
    data = dice_ml.Data(
        dataframe=dataframe,
        continuous_features=continuous_columns,
        outcome_name=outcome_name,
        permitted_range=categorical_ranges,
    )

    dice_model = (
        _RecodingWrapper(model, categorical_dtypes)
        if categorical_dtypes
        else model
    )
    explainer = dice_ml.Dice(
        data,
        dice_ml.Model(model=dice_model, backend="sklearn"),
        method="random",
    )

    query_model_ready = X_encoded.iloc[0:1]
    prohibited_features = set(policy["excluded_sensitive_features"])
    prohibited_features.update(policy["excluded_immutable_features"])
    generation_options = {
        "total_CFs": 2,
        "desired_class": "opposite",
    }
    generation_options["features_to_vary"] = permitted_features

    dice_exp = explainer.generate_counterfactuals(
        query_model_ready,
        **generation_options,
    )
    counterfactual_frame = _extract_counterfactual_frame(dice_exp, outcome_name)

    query_raw = X_raw.iloc[0]
    query_ready = query_model_ready.iloc[0]
    original_prediction, original_probability = _prediction(
        model, X_raw.iloc[0:1]
    )

    structured_counterfactuals = []
    all_decoding_reliable = True
    sensitive_change_detected = False
    policy_violation_detected = False
    for index, (_, counterfactual_ready) in enumerate(
        counterfactual_frame.iterrows(), 1
    ):
        counterfactual_input = counterfactual_ready.to_frame().T
        counterfactual_prediction, counterfactual_probability = _prediction(
            dice_model, counterfactual_input
        )
        changes, decoding_reliable = _changed_features(
            query_raw,
            query_ready,
            counterfactual_ready,
            categorical_dtypes,
            prohibited_features,
        )
        all_decoding_reliable = all_decoding_reliable and decoding_reliable
        sensitive_change_detected = sensitive_change_detected or any(
            change["is_sensitive_or_immutable"] for change in changes
        )
        policy_violation_detected = policy_violation_detected or any(
            change["feature"] not in permitted_features for change in changes
        )
        display_values = {}
        for feature, value in counterfactual_ready.items():
            display_values[feature], decoded = _decode_value(
                feature, value, categorical_dtypes
            )
            all_decoding_reliable = all_decoding_reliable and decoded

        structured_counterfactuals.append(
            {
                "counterfactual_id": index,
                "counterfactual_prediction": counterfactual_prediction,
                "counterfactual_prediction_proba": counterfactual_probability,
                "values": display_values,
                "model_ready_values": {
                    feature: _json_value(value)
                    for feature, value in counterfactual_ready.items()
                },
                "changed_features": changes,
                "number_of_changed_features": len(changes),
            }
        )

    limitations = [
        "Counterfactual examples are generated mathematically and may not be feasible or actionable in the real world.",
        "Counterfactual examples should not recommend changes to immutable or protected attributes.",
        "Domain experts should review counterfactual changes before using them for decision support.",
    ]
    if not all_decoding_reliable:
        limitations.append(
            "Some categorical values may reflect model-ready encoded representations."
        )
    if sensitive_change_detected:
        limitations.append(
            "The generated counterfactual changes potentially immutable or sensitive "
            "features and should not be treated as an actionable recommendation."
        )
    if policy["counterfactual_policy_status"] == "metadata_not_provided":
        limitations.append(
            "The audit context did not provide actionable or immutable feature "
            "metadata; generated "
            "changes must not be described as actionable recommendations."
        )

    unique_changed_features = list(
        dict.fromkeys(
            change["feature"]
            for item in structured_counterfactuals
            for change in item["changed_features"]
        )
    )
    first_counterfactual = structured_counterfactuals[0]
    artifact = {
        "query_instance": {
            feature: _json_value(value) for feature, value in query_raw.items()
        },
        "query_instance_model_ready": {
            feature: _json_value(value) for feature, value in query_ready.items()
        },
        "original_prediction": original_prediction,
        "original_prediction_proba": original_probability,
        "desired_class": "opposite",
        "counterfactual_prediction": first_counterfactual[
            "counterfactual_prediction"
        ],
        "counterfactual_prediction_proba": first_counterfactual[
            "counterfactual_prediction_proba"
        ],
        "counterfactuals": structured_counterfactuals,
        "counterfactuals_count": len(structured_counterfactuals),
        "changed_features": first_counterfactual["changed_features"],
        "changed_features_summary": {
            "first_counterfactual_count": first_counterfactual[
                "number_of_changed_features"
            ],
            "maximum_per_counterfactual": max(
                item["number_of_changed_features"]
                for item in structured_counterfactuals
            ),
            "unique_changed_features": unique_changed_features,
            "sensitive_or_immutable_change_detected": sensitive_change_detected,
        },
        "generation_method": "DiCE random",
        **policy,
        "generation_dtype_schema": {
            feature: str(dtype) for feature, dtype in X_encoded.dtypes.items()
        },
        "permitted_categorical_values": categorical_ranges,
        "policy_violation_detected": policy_violation_detected,
        "sensitive_or_immutable_change_detected": sensitive_change_detected,
        "protected_features_requested": policy["excluded_sensitive_features"],
        "features_permitted_to_vary": permitted_features,
        "limitations": limitations,
    }

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(artifact, output_file, indent=2, ensure_ascii=False, allow_nan=False)

    return {
        "type": "explainability",
        "method": "DiCE",
        "evidence_type": "counterfactual_explanation",
        "output": output_path,
        **artifact,
    }
