"""
Fairness analysis module using Fairlearn.
"""

from fairlearn.metrics import demographic_parity_difference


def run_fairness(model, X, y):

    predictions = model.predict(X)

    # Example sensitive attribute (placeholder)
    sensitive_feature = [i % 2 for i in range(len(X))]

    dp = demographic_parity_difference(
        y_true=y,
        y_pred=predictions,
        sensitive_features=sensitive_feature
    )

    print("Demographic parity difference:", dp)
