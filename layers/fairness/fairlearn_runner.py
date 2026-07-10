"""
Fairness analysis module using Fairlearn.
"""


def run_fairness(model, X, y, sensitive_features=None):
    """Compute fairness metrics for each sensitive feature column.

    Parameters
    ----------
    sensitive_features : list[str] | None
        Column names in X that are considered sensitive attributes.
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

    preds = model.predict(X)
    per_feature = {}

    for sf_name in sensitive_features:
        if sf_name not in X.columns:
            per_feature[sf_name] = {
                "error": f"Column '{sf_name}' not found in feature matrix."
            }
            continue

        sf_col = X[sf_name]

        mf = MetricFrame(
            metrics={"selection_rate": selection_rate},
            y_true=y,
            y_pred=preds,
            sensitive_features=sf_col,
        )

        dp_diff = demographic_parity_difference(
            y_true=y, y_pred=preds, sensitive_features=sf_col
        )

        eo_diff = round(
            float(equalized_odds_difference(
                y_true=y, y_pred=preds, sensitive_features=sf_col
            )),
            4,
        )

        by_group = mf.by_group["selection_rate"].to_dict()
        selection_rates = {
            f"selection_rate_{group}": round(float(rate), 4)
            for group, rate in sorted(by_group.items(), key=lambda kv: str(kv[0]))
        }

        per_feature[sf_name] = {
            "demographic_parity_difference": round(float(dp_diff), 4),
            "equalized_odds_difference": eo_diff,
            **selection_rates,
        }

        print(f"[Fairness] {sf_name}: dp_diff={dp_diff:.4f}")

    return {
        "type": "fairness",
        "method": "Fairlearn",
        **per_feature,
    }

    
