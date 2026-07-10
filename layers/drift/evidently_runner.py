"""
Dataset drift detection using Evidently.
The drift detection layer uses Evidently AI to generate automated reports
that highlight distribution shifts between reference and current datasets.
"""

import os


def run_drift(X):
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise ImportError(
            "Evidently is required to run drift analysis."
        ) from exc

    os.makedirs("outputs/drift", exist_ok=True)

    # Simulate reference vs current（demo）
    split = int(len(X) * 0.5)

    reference_data = X[:split]
    current_data = X[split:]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    report_metric = None
    if hasattr(report, "metrics") and report.metrics:
        report_metric = report.metrics[0]

    metric_dict = report_metric.dict() if report_metric is not None else {}
    drift_share = metric_dict.get("drift_share")
    drift_share = round(float(drift_share), 4) if drift_share is not None else None
    features_analyzed = len(X.columns) if hasattr(X, "columns") else None
    drifted_features = []

    result = {
        "type": "drift",
        "method": "Evidently",
        "drift_share": drift_share,
        "features_analyzed": features_analyzed,
        "drifted_features": drifted_features,
        "note": (
            "Current Evidently API exposes dataset-level drift_share via the "
            "preset metric object; per-column drift details are not returned "
            "by this version."
        ),
    }

    return result
