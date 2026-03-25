"""
Dataset drift detection using Evidently.
"""

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def run_drift(X):

    reference_data = pd.DataFrame(X)
    current_data = pd.DataFrame(X.sample(frac=1))

    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    print("Drift analysis completed.")
