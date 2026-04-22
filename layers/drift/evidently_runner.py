"""
Dataset drift detection using Evidently.
The drift detection layer uses Evidently AI to generate automated reports
that highlight distribution shifts between reference and current datasets.
"""

import os
from evidently import Report
from evidently.presets import DataDriftPreset


def run_drift(X):

    os.makedirs("outputs/drift", exist_ok=True)

    # Simulate reference vs current（demo）
    split = int(len(X) * 0.5)

    reference_data = X[:split]
    current_data = X[split:]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    items_callable = getattr(report, "items", None)
    if callable(items_callable):
        metrics_snapshot = [str(item) for item in items_callable()]
    else:
        metrics_snapshot = ["Drift detection completed"]

    result = {
        "type": "drift",
        "method": "Evidently",
        "metrics": metrics_snapshot,
    }

    return result
