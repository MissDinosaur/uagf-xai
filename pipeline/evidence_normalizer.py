"""Normalize evidence runner outputs without changing evidence algorithms."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from layers.llm.evidence_methods import (
    LLM_GROUNDING,
    LLM_PROMPT_FAIRNESS,
    LLM_SELF_CONSISTENCY,
    LLM_SEMANTIC_DRIFT,
    llm_method_display_name,
)
from schema.evidence_schema import (
    EVIDENCE_STATUSES,
    completed_evidence,
    failed_evidence,
    not_applicable_evidence,
    skipped_evidence,
)


EVIDENCE_CATALOG = {
    "shap": {
        "result_key": "explainability",
        "evidence_id": "EXP-SHAP",
        "layer": "explainability",
        "method": "SHAP",
        "articles": ["Art. 13"],
    },
    "lime": {
        "result_key": "lime",
        "evidence_id": "EXP-LIME",
        "layer": "explainability",
        "method": "LIME",
        "articles": ["Art. 13"],
    },
    "dice": {
        "result_key": "counterfactual",
        "evidence_id": "EXP-DICE",
        "layer": "counterfactual",
        "method": "DiCE",
        "articles": ["Art. 13"],
    },
    "fairness": {
        "result_key": "fairness",
        "evidence_id": "FAIR-FAIRLEARN",
        "layer": "fairness",
        "method": "Fairlearn",
        "articles": ["Art. 10"],
    },
    "uncertainty": {
        "result_key": "uncertainty",
        "evidence_id": "UNC-MAPIE",
        "layer": "uncertainty",
        "method": "MAPIE (Conformal Prediction)",
        "articles": ["Art. 14", "Art. 15"],
    },
    "drift": {
        "result_key": "drift",
        "evidence_id": "DRIFT-EVIDENTLY",
        "layer": "drift",
        "method": "Evidently + Feature Drift Tests",
        "articles": ["Art. 15", "Art. 61"],
    },
    LLM_GROUNDING: {
        "result_key": LLM_GROUNDING,
        "evidence_id": "LLM-E1",
        "layer": "llm_explainability",
        "method": llm_method_display_name(LLM_GROUNDING),
        "articles": ["Art. 13"],
    },
    LLM_SELF_CONSISTENCY: {
        "result_key": LLM_SELF_CONSISTENCY,
        "evidence_id": "LLM-E2",
        "layer": "llm_uncertainty",
        "method": llm_method_display_name(LLM_SELF_CONSISTENCY),
        "articles": ["Art. 15"],
    },
    LLM_SEMANTIC_DRIFT: {
        "result_key": LLM_SEMANTIC_DRIFT,
        "evidence_id": "LLM-E3",
        "layer": "llm_drift",
        "method": llm_method_display_name(LLM_SEMANTIC_DRIFT),
        "articles": ["Art. 61"],
    },
    LLM_PROMPT_FAIRNESS: {
        "result_key": LLM_PROMPT_FAIRNESS,
        "evidence_id": "LLM-E4",
        "layer": "llm_fairness",
        "method": llm_method_display_name(LLM_PROMPT_FAIRNESS),
        "articles": ["Art. 10"],
    },
}

RESULT_KEY_TO_TOKEN = {
    details["result_key"]: token for token, details in EVIDENCE_CATALOG.items()
}

REQUIRED_FIELDS = frozenset(
    {
        "evidence_id",
        "layer",
        "method",
        "status",
        "article_mapping",
        "summary",
        "key_findings",
        "metrics",
        "artifacts",
        "limitations",
        "raw_output",
    }
)


def is_unified_evidence(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and REQUIRED_FIELDS.issubset(value)
        and value.get("status") in EVIDENCE_STATUSES
    )


def _status(raw: dict) -> str:
    status = str(raw.get("status") or "completed").strip().lower()
    if status in EVIDENCE_STATUSES:
        return status
    if raw.get("error"):
        return "failed"
    return "completed"


def _artifact_paths(raw: dict) -> list[str]:
    paths = []
    for key in ("plot", "output"):
        value = raw.get(key)
        if value:
            paths.append(str(value))
    return paths


def _base_payload(token: str, raw: dict) -> dict[str, Any]:
    catalog = EVIDENCE_CATALOG[token]
    return {
        "evidence_id": catalog["evidence_id"],
        "layer": catalog["layer"],
        "method": catalog["method"],
        "article_mapping": catalog["articles"],
        "artifacts": _artifact_paths(raw),
        "raw_output": deepcopy(raw),
    }


def _skipped_or_failed(token: str, raw: dict) -> dict[str, Any] | None:
    status = _status(raw)
    if status not in {"skipped", "failed"}:
        return None
    catalog = EVIDENCE_CATALOG[token]
    reason = str(raw.get("reason") or raw.get("error") or "No reason was provided.")
    factory = skipped_evidence if status == "skipped" else failed_evidence
    return factory(
        **_base_payload(token, raw),
        summary=f"{catalog['method']} {status}: {reason}",
        key_findings=[],
        metrics={},
        limitations=[reason],
    )


def _normalize_shap(raw: dict) -> dict:
    top_features = list(raw.get("top_features") or [])
    findings = []
    if top_features:
        findings.append(f"The top-ranked feature is {top_features[0]}.")
        findings.append(
            "The top features are " + ", ".join(map(str, top_features[:3])) + "."
        )
    return completed_evidence(
        **_base_payload("shap", raw),
        summary="SHAP identified the most influential features driving model predictions.",
        key_findings=findings,
        metrics={
            "explainer": raw.get("explainer"),
            "top_features": top_features,
            "feature_importance": deepcopy(raw.get("feature_importance") or []),
        },
        limitations=[],
    )


def _normalize_lime(raw: dict) -> dict:
    fidelity = raw.get("local_fidelity_score")
    limitations = [
        "LIME is a local surrogate explanation for one sample and does not establish global model behaviour.",
        "Categorical thresholds may refer to saved encoder values in the model-ready numeric representation.",
    ]
    if isinstance(fidelity, (int, float)) and fidelity < 0.5:
        limitations.append(
            f"The local surrogate fidelity score is {fidelity}, indicating a weak local approximation."
        )
    return completed_evidence(
        **_base_payload("lime", raw),
        summary="LIME generated a local surrogate explanation for one evaluation sample.",
        key_findings=[
            f"{raw.get('features_explained', 0)} local feature contributions were produced.",
            f"The local fidelity score is {fidelity}.",
        ],
        metrics={
            "mode": raw.get("mode"),
            "sample_index": raw.get("sample_index"),
            "features_explained": raw.get("features_explained"),
            "feature_contributions": deepcopy(raw.get("feature_contributions") or []),
            "local_fidelity_score": fidelity,
            "local_prediction": raw.get("local_prediction"),
            "prediction": deepcopy(raw.get("prediction") or {}),
            "input_representation": raw.get("input_representation"),
        },
        limitations=limitations,
    )


def _normalize_fairness(raw: dict) -> dict:
    ignored = {"type", "method", "status", "reason"}
    per_feature = {
        key: deepcopy(value)
        for key, value in raw.items()
        if key not in ignored and isinstance(value, dict)
    }
    dp_values = [
        (name, value.get("demographic_parity_difference"))
        for name, value in per_feature.items()
        if isinstance(value.get("demographic_parity_difference"), (int, float))
    ]
    eo_values = [
        (name, value.get("equalized_odds_difference"))
        for name, value in per_feature.items()
        if isinstance(value.get("equalized_odds_difference"), (int, float))
    ]
    findings = []
    if dp_values:
        feature, value = max(dp_values, key=lambda item: item[1])
        findings.append(
            f"The largest demographic parity difference was observed for {feature} ({value})."
        )
    if eo_values:
        feature, value = max(eo_values, key=lambda item: item[1])
        findings.append(
            f"The largest equalized odds difference was observed for {feature} ({value})."
        )
    limitations = [
        "Fairness metrics indicate statistical group differences but do not by themselves prove unlawful discrimination."
    ]
    if any("age" in name.lower() for name in per_feature):
        limitations.append(
            "Age was evaluated as individual values; grouped age bands would be more audit-friendly."
        )
    return completed_evidence(
        **_base_payload("fairness", raw),
        summary="Fairlearn computed group fairness metrics for the configured sensitive features.",
        key_findings=findings,
        metrics={
            "sensitive_features": list(per_feature),
            "per_feature": per_feature,
        },
        limitations=limitations,
    )


def _normalize_uncertainty(raw: dict) -> dict:
    metrics = {
        key: raw.get(key)
        for key in (
            "confidence_level",
            "coverage",
            "mean_interval_width",
            "coverage_gap",
        )
    }
    return completed_evidence(
        **_base_payload("uncertainty", raw),
        summary="MAPIE generated conformal uncertainty evidence for model predictions.",
        key_findings=[
            f"Observed coverage is {metrics['coverage']} at confidence level {metrics['confidence_level']}.",
            f"The coverage gap is {metrics['coverage_gap']}.",
        ],
        metrics=metrics,
        limitations=[],
    )


def _normalize_drift(raw: dict) -> dict:
    limitations = []
    if raw.get("note"):
        limitations.append(str(raw["note"]))
    return completed_evidence(
        **_base_payload("drift", raw),
        summary="Evidently produced dataset-level drift monitoring evidence.",
        key_findings=[
            f"Dataset-level drift share is {raw.get('drift_share')}.",
            f"{raw.get('features_analyzed')} features were analyzed.",
        ],
        metrics={
            "drift_share": raw.get("drift_share"),
            "features_analyzed": raw.get("features_analyzed"),
            "drifted_features": deepcopy(raw.get("drifted_features") or []),
        },
        limitations=limitations,
    )


def _normalize_dice(raw: dict) -> dict:
    count = raw.get("counterfactuals_count")
    return completed_evidence(
        **_base_payload("dice", raw),
        summary="DiCE generated counterfactual examples for local decision explanation.",
        key_findings=[
            f"{count} counterfactual result set(s) were generated.",
            f"The artifact was saved to {raw.get('output')}.",
        ],
        metrics={"counterfactuals_count": count},
        limitations=[
            "Counterfactual examples should be reviewed for feasibility and domain validity."
        ],
    )


def _normalize_llm(token: str, raw: dict) -> dict:
    catalog = EVIDENCE_CATALOG[token]
    status = _status(raw)
    available = list(raw.get("available_resources") or [])
    if status == "skipped":
        findings = [f"{resource.replace('_', ' ').title()} was available." for resource in available]
        findings.append("No loadable model weights were found.")
        return skipped_evidence(
            **_base_payload(token, raw),
            summary=(
                f"{catalog['method']} was selected but could not be executed "
                "because the LLM artifact is metadata-only."
            ),
            key_findings=findings,
            metrics={},
            limitations=[
                "Execution-level LLM evidence requires a complete local HuggingFace model artifact."
            ],
        )

    metrics = {
        key: deepcopy(value)
        for key, value in raw.items()
        if key not in {"type", "method", "status", "reason"}
    }
    return completed_evidence(
        **_base_payload(token, raw),
        summary=f"{catalog['method']} completed and produced structured LLM evidence.",
        key_findings=[],
        metrics=metrics,
        limitations=[],
    )


NORMALIZERS = {
    "shap": _normalize_shap,
    "lime": _normalize_lime,
    "fairness": _normalize_fairness,
    "uncertainty": _normalize_uncertainty,
    "drift": _normalize_drift,
    "dice": _normalize_dice,
}


def _compatibility_reason(token: str, task_type: str) -> str:
    method = EVIDENCE_CATALOG[token]["method"]
    reasons = {
        ("forecasting", "fairness"): "Forecasting is not a group classification task in the current implementation.",
        ("forecasting", "lime"): "The current LIME pathway is not enabled for forecasting.",
        ("forecasting", "dice"): "The current DiCE pathway is not applicable to the forecasting wrapper.",
        ("forecasting", "uncertainty"): "The current MAPIE runner does not support the forecasting wrapper.",
        ("anomaly_detection", "fairness"): "The current fairness runner supports classification group metrics, not anomaly scores.",
        ("anomaly_detection", "lime"): "The current LIME pathway is not enabled for anomaly detection.",
        ("anomaly_detection", "dice"): "IsolationForest-style anomaly detection does not expose standard class probabilities required by DiCE.",
        ("anomaly_detection", "uncertainty"): "The current MAPIE runner is not compatible with IsolationForest-style anomaly detection.",
    }
    return reasons.get(
        (task_type, token),
        f"{method} is not applicable to task type {task_type} in the current implementation.",
    )


def _not_applicable_result(token: str, task_type: str, trace: dict) -> dict:
    catalog = EVIDENCE_CATALOG[token]
    reason = _compatibility_reason(token, task_type)
    return not_applicable_evidence(
        evidence_id=catalog["evidence_id"],
        layer=catalog["layer"],
        method=catalog["method"],
        article_mapping=catalog["articles"],
        summary=f"{catalog['method']} was not executed because it is not applicable to {task_type}.",
        key_findings=[],
        metrics={},
        artifacts=[],
        limitations=[reason],
        raw_output={
            "task_type": task_type,
            "compatibility_assessment": deepcopy(trace),
        },
    )


def normalize_evidence_results(
    results: dict,
    *,
    cbep_trace: dict | None = None,
    task_type: str | None = None,
) -> dict:
    """Return evidence results in the unified schema while preserving keys."""
    normalized: dict[str, Any] = {}

    for result_key, value in results.items():
        if result_key == "_cbep_trace":
            normalized[result_key] = deepcopy(value)
            continue

        token = RESULT_KEY_TO_TOKEN.get(result_key)
        if token is None or not isinstance(value, dict):
            normalized[result_key] = deepcopy(value)
            continue
        if is_unified_evidence(value):
            normalized[result_key] = deepcopy(value)
            continue

        special_status = _skipped_or_failed(token, value)
        if special_status is not None and not token.startswith("llm_"):
            normalized[result_key] = special_status
        elif token.startswith("llm_"):
            normalized[result_key] = _normalize_llm(token, value)
        else:
            normalized[result_key] = NORMALIZERS[token](value)

    trace = cbep_trace or normalized.get("_cbep_trace") or {}
    assessment = trace.get("task_compatibility_assessment", {}) if isinstance(trace, dict) else {}
    incompatible = assessment.get("incompatible_methods", []) if isinstance(assessment, dict) else []
    normalized_task = str(task_type or trace.get("task_type") or "unknown")

    for token in incompatible:
        catalog = EVIDENCE_CATALOG.get(token)
        if not catalog:
            continue
        result_key = catalog["result_key"]
        if result_key not in normalized:
            normalized[result_key] = _not_applicable_result(
                token,
                normalized_task,
                assessment,
            )

    selected = trace.get("final_plan", []) if isinstance(trace, dict) else []
    for token in selected:
        catalog = EVIDENCE_CATALOG.get(token)
        if not catalog:
            continue
        result_key = catalog["result_key"]
        if result_key in normalized:
            continue
        reason = (
            "CBEP selected this method, but the executor returned no evidence result."
        )
        normalized[result_key] = failed_evidence(
            evidence_id=catalog["evidence_id"],
            layer=catalog["layer"],
            method=catalog["method"],
            article_mapping=catalog["articles"],
            summary=reason,
            key_findings=[],
            metrics={},
            artifacts=[],
            limitations=[reason],
            raw_output={},
        )

    return normalized
