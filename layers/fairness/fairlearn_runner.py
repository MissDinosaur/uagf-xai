"""
Fairness analysis module using Fairlearn.
"""

import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference

def run_fairness(model, X, y):

    # The first column is sensitive feature for demo purposes
    sensitive_feature = X.iloc[:, 0]

    preds = model.predict(X)

    metrics = {
        "selection_rate": selection_rate
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y,
        y_pred=preds,
        sensitive_features=sensitive_feature
    )

    dp_diff = demographic_parity_difference(
        y_true=y,
        y_pred=preds,
        sensitive_features=sensitive_feature
    )

    result = {
        "type": "fairness",
        "selection_rate_by_group": mf.by_group.to_dict(),
        "demographic_parity_difference": float(dp_diff)
    }
    print("Demographic parity difference:", dp_diff)

    return result

    
