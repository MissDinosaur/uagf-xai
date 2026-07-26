"""Traditional ML dataset drift evidence using raw reference/current data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from output_naming import build_output_path
from schema.evidence_schema import completed_evidence, skipped_evidence


EVIDENCE_ID = "DRIFT-EVIDENTLY"
METHOD_NAME = "Evidently + Feature Drift Tests"
ARTICLE_MAPPING = ["Art. 15", "Art. 61"]


def _as_dataframe(value: Any) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value)
    except (TypeError, ValueError):
        return None


def _is_obvious_label(column: Any, target_column: str | None) -> bool:
    name = str(column).strip().lower()
    target = str(target_column).strip().lower() if target_column else None
    return bool(
        (target and name == target)
        or name in {"target", "label", "labels", "outcome", "y"}
        or name.endswith("_label")
    )


def _numeric_test(reference: pd.Series, current: pd.Series) -> dict[str, Any]:
    reference_values = pd.to_numeric(reference, errors="coerce").dropna()
    current_values = pd.to_numeric(current, errors="coerce").dropna()
    if reference_values.empty or current_values.empty:
        raise ValueError("No non-null numeric observations were available in both datasets.")

    statistic, p_value = ks_2samp(reference_values, current_values)
    return {
        "feature_type": "numeric",
        "method": "ks_test",
        "statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 6),
    }


def _categorical_test(reference: pd.Series, current: pd.Series) -> dict[str, Any]:
    missing = "__UAGF_MISSING__"
    reference_values = reference.astype("string").fillna(missing)
    current_values = current.astype("string").fillna(missing)
    categories = sorted(set(reference_values.unique()) | set(current_values.unique()))
    if not categories:
        raise ValueError("No categorical observations were available in either dataset.")

    reference_counts = reference_values.value_counts(normalize=True).reindex(
        categories, fill_value=0.0
    )
    current_counts = current_values.value_counts(normalize=True).reindex(
        categories, fill_value=0.0
    )
    statistic = jensenshannon(
        reference_counts.to_numpy(dtype=float),
        current_counts.to_numpy(dtype=float),
        base=2,
    )
    return {
        "feature_type": "categorical",
        "method": "jensen_shannon",
        "statistic": round(float(statistic), 6),
        "p_value": None,
    }


def _run_evidently(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[Any, str | None]:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise ImportError("Evidently is required to run drift analysis.") from exc

    try:
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference, current_data=current)
        if not hasattr(snapshot, "dump_dict"):
            return {}, "Evidently did not expose a serializable report snapshot."

        snapshot_data = snapshot.dump_dict()
        metric_results = {}
        for metric_id, metric in snapshot_data.get("metric_results", {}).items():
            if isinstance(metric, dict):
                metric_results[metric_id] = {
                    key: value
                    for key, value in metric.items()
                    if key not in {"widget", "tests"}
                }
        compact_snapshot = {
            "name": snapshot_data.get("name"),
            "timestamp": snapshot_data.get("timestamp"),
            "metadata": snapshot_data.get("metadata", {}),
            "tags": snapshot_data.get("tags", []),
            "top_level_metrics": snapshot_data.get("top_level_metrics", []),
            "metric_results": metric_results,
        }
        return json.loads(json.dumps(compact_snapshot, default=str)), None
    except Exception as exc:  # Evidently API/data compatibility varies by release.
        return {}, f"Evidently execution was unavailable: {type(exc).__name__}: {exc}"


def _save_feature_tests(
    feature_tests: list[dict[str, Any]],
    provider_name: str | None,
    output_namespace: str,
) -> str:
    output_path = build_output_path(
        "drift",
        provider_name,
        "feature_drift",
        ".json",
        fallback=output_namespace or "audit",
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(feature_tests, handle, indent=2, ensure_ascii=False)
    return output_path


def run_drift(
    current_data,
    *,
    reference_data=None,
    target_column: str | None = None,
    provider_name: str | None = None,
    output_namespace: str = "audit",
    numeric_threshold: float = 0.05,
    categorical_threshold: float = 0.1,
    dataset_drift_threshold: float = 0.2,
):
    """Compare raw training/reference data with raw evaluation/current data."""
    reference_df = _as_dataframe(reference_data)
    current_df = _as_dataframe(current_data)
    if reference_df is None or current_df is None:
        reason = "Reference and current datasets are required for drift analysis."
        return skipped_evidence(
            evidence_id=EVIDENCE_ID,
            layer="drift",
            method=METHOD_NAME,
            article_mapping=ARTICLE_MAPPING,
            summary="Drift analysis was skipped because reference or current data was unavailable.",
            key_findings=[],
            metrics={},
            artifacts=[],
            limitations=[reason],
            raw_output={},
        )

    reference_columns = set(reference_df.columns)
    current_columns = set(current_df.columns)
    all_columns = reference_columns | current_columns
    excluded_columns = sorted(
        [column for column in all_columns if _is_obvious_label(column, target_column)],
        key=str,
    )
    common_columns = sorted(reference_columns & current_columns, key=str)
    feature_columns = [column for column in common_columns if column not in excluded_columns]
    reference_features = reference_df.loc[:, feature_columns]
    current_features = current_df.loc[:, feature_columns]

    feature_tests: list[dict[str, Any]] = []
    skipped_columns: list[dict[str, str]] = []
    for column in feature_columns:
        reference_series = reference_features[column]
        current_series = current_features[column]
        try:
            if (
                pd.api.types.is_datetime64_any_dtype(reference_series)
                or pd.api.types.is_datetime64_any_dtype(current_series)
            ):
                raise ValueError("Datetime columns are not supported by the current fallback tests.")
            if pd.api.types.is_numeric_dtype(reference_series) and pd.api.types.is_numeric_dtype(current_series):
                test = _numeric_test(reference_series, current_series)
                detected = test["p_value"] < numeric_threshold
            else:
                test = _categorical_test(reference_series, current_series)
                detected = test["statistic"] >= categorical_threshold
            feature_tests.append(
                {"feature": str(column), **test, "drift_detected": bool(detected)}
            )
        except (TypeError, ValueError) as exc:
            skipped_columns.append({"feature": str(column), "reason": str(exc)})

    if not feature_tests:
        reason = "No common supported feature columns could be evaluated for drift."
        return skipped_evidence(
            evidence_id=EVIDENCE_ID,
            layer="drift",
            method=METHOD_NAME,
            article_mapping=ARTICLE_MAPPING,
            summary="Drift analysis was skipped because no comparable features were available.",
            key_findings=[],
            metrics={},
            artifacts=[],
            limitations=[reason],
            raw_output={
                "common_columns": [str(column) for column in common_columns],
                "excluded_columns": [str(column) for column in excluded_columns],
                "skipped_columns": skipped_columns,
            },
        )

    evidently_result, evidently_limitation = _run_evidently(
        reference_features,
        current_features,
    )
    ranked_tests = sorted(
        feature_tests,
        key=lambda item: (bool(item["drift_detected"]), float(item["statistic"])),
        reverse=True,
    )
    drifted_features = [
        item["feature"] for item in ranked_tests if item["drift_detected"]
    ]
    features_analyzed = len(feature_tests)
    drift_share = round(len(drifted_features) / features_analyzed, 4)
    dataset_drift_detected = drift_share >= dataset_drift_threshold

    limitations = [
        "Feature-level statistical drift does not prove model performance degradation.",
        "Drift results depend on sample size and chosen statistical thresholds.",
        "Categorical drift uses distribution distance and should be interpreted with domain knowledge.",
        "The current Evidently API version may not expose per-column details, so UAGF-XAI adds fallback feature-level tests.",
    ]
    if evidently_limitation:
        limitations.append(evidently_limitation)
    if skipped_columns:
        limitations.append(
            f"{len(skipped_columns)} common column(s) could not be evaluated; details are preserved in raw_output."
        )
    reference_only = sorted(
        (reference_columns - current_columns) - set(excluded_columns), key=str
    )
    current_only = sorted(
        (current_columns - reference_columns) - set(excluded_columns), key=str
    )
    if reference_only or current_only:
        limitations.append(
            "Columns present in only one dataset were excluded from comparison."
        )

    method_details = {
        "primary_engine": "Evidently",
        "fallback_feature_tests": {
            "numeric": "scipy.stats.ks_2samp",
            "categorical": "scipy.spatial.distance.jensenshannon",
        },
        "numeric_p_value_threshold": numeric_threshold,
        "categorical_distance_threshold": categorical_threshold,
        "dataset_drift_share_threshold": dataset_drift_threshold,
        "evidently_result_available": bool(evidently_result),
    }
    artifact_path = _save_feature_tests(
        ranked_tests,
        provider_name,
        output_namespace,
    )
    metrics = {
        "dataset_drift_detected": bool(dataset_drift_detected),
        "drift_share": drift_share,
        "features_analyzed": features_analyzed,
        "number_of_columns": features_analyzed,
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "drifted_features": drifted_features,
        "top_drifted_columns": ranked_tests[:10],
        "method_details": method_details,
    }
    decision_text = "detected" if dataset_drift_detected else "not detected"
    summary = (
        f"The drift analysis compared {len(reference_df)} reference rows against "
        f"{len(current_df)} current rows across {features_analyzed} common feature columns. "
        f"{len(drifted_features)} feature(s) were flagged as drifted, producing a drift "
        f"share of {drift_share:.4f}. Under the configured dataset-level threshold of "
        f"{dataset_drift_threshold:.2f}, dataset-level drift was {decision_text}."
    )
    key_findings = [
        f"{features_analyzed} features were analyzed.",
        f"{len(drifted_features)} features were flagged as drifted.",
        f"Dataset-level drift share is {drift_share:.4f}.",
        f"Dataset-level drift was {decision_text} under the configured threshold.",
    ]
    if dataset_drift_detected:
        key_findings.append(
            "The most drifted features should be reviewed before relying on current model predictions."
        )

    return completed_evidence(
        evidence_id=EVIDENCE_ID,
        layer="drift",
        method=METHOD_NAME,
        article_mapping=ARTICLE_MAPPING,
        summary=summary,
        key_findings=key_findings,
        metrics=metrics,
        artifacts=[artifact_path],
        limitations=limitations,
        raw_output={
            "evidently_result": evidently_result,
            "feature_tests": ranked_tests,
            "thresholds": {
                "numeric_p_value": numeric_threshold,
                "categorical_distance": categorical_threshold,
                "dataset_drift_share": dataset_drift_threshold,
            },
            "common_columns": [str(column) for column in feature_columns],
            "excluded_columns": [str(column) for column in excluded_columns],
            "reference_only_columns": [str(column) for column in reference_only],
            "current_only_columns": [str(column) for column in current_only],
            "skipped_columns": skipped_columns,
        },
    )
