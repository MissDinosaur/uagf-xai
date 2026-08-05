"""
Fairness analysis module using Fairlearn.
"""

from functools import partial

import numpy as np


def run_fairness(
    model,
    X_model,
    y,
    *,
    sensitive_data=None,
    sensitive_features=None,
    positive_label=1,
):
    """Compute fairness metrics for each sensitive feature column.

    Parameters
    ----------
    sensitive_features : list[str] | None
        Column names in sensitive_data that are considered sensitive attributes.
        If empty or None, the fairness step is skipped.
    """
    try:
        from fairlearn.metrics import (
            MetricFrame,
            selection_rate,
            demographic_parity_difference,
            equalized_odds_difference,
        )
    except ImportError as exc:
        raise ImportError(
            "Fairlearn is required to run fairness analysis."
        ) from exc

    if not sensitive_features:
        raise ValueError(
            "Fairness analysis requires at least one sensitive feature column."
        )

    if y is None:
        raise ValueError("Fairness analysis requires evaluation labels.")
    group_frame = sensitive_data if sensitive_data is not None else X_model
    if not hasattr(group_frame, "columns"):
        raise TypeError("Fairness sensitive data must be a pandas DataFrame.")
    if len(X_model) != len(y) or len(group_frame) != len(y):
        raise ValueError(
            "Model input, labels, and sensitive data must have equal row counts."
        )
    if (
        hasattr(X_model, "index")
        and hasattr(y, "index")
        and not X_model.index.equals(y.index)
    ):
        raise ValueError("Model input and labels must preserve aligned indexes.")
    if hasattr(group_frame, "index") and hasattr(y, "index"):
        if not group_frame.index.equals(y.index):
            raise ValueError("Sensitive data and labels must preserve aligned indexes.")

    preds = model.predict(X_model)
    effective_positive_label = 1 if positive_label is None else positive_label
    binary_predictions = np.asarray(preds) == effective_positive_label
    binary_labels = np.asarray(y) == effective_positive_label
    per_feature = {}

    for sf_name in sensitive_features:
        if sf_name not in group_frame.columns:
            per_feature[sf_name] = {
                "error": f"Column '{sf_name}' not found in sensitive data.",
            }
            continue

        try:
            sf_col = group_frame[sf_name]
            metric = partial(selection_rate, pos_label=True)
            mf = MetricFrame(
                metrics={"selection_rate": metric},
                y_true=binary_labels,
                y_pred=binary_predictions,
                sensitive_features=sf_col,
            )
            dp_diff = demographic_parity_difference(
                y_true=binary_labels,
                y_pred=binary_predictions,
                sensitive_features=sf_col,
            )
            eo_diff = equalized_odds_difference(
                y_true=binary_labels,
                y_pred=binary_predictions,
                sensitive_features=sf_col,
            )
            by_group = mf.by_group["selection_rate"].to_dict()
            selection_rates = {
                f"selection_rate_{group}": round(float(rate), 4)
                for group, rate in sorted(by_group.items(), key=lambda item: str(item[0]))
            }
            per_feature[sf_name] = {
                "demographic_parity_difference": round(float(dp_diff), 4),
                "equalized_odds_difference": round(float(eo_diff), 4),
                **selection_rates,
            }
            print(f"[Fairness] {sf_name}: dp_diff={dp_diff:.4f}")
        except Exception as exc:
            per_feature[sf_name] = {
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {
        "type": "fairness",
        "method": "Fairlearn",
        "status": "completed",
        "positive_label": effective_positive_label,
        **per_feature,
    }

    
