"""Build a presentation-oriented view model for the HTML audit report."""

from __future__ import annotations

from typing import Any

from layers.llm.evidence_methods import (
    LLM_GROUNDING,
    LLM_PROMPT_FAIRNESS,
    LLM_SELF_CONSISTENCY,
    LLM_SEMANTIC_DRIFT,
)
from report.method_ordering import (
    LLM_METHOD_ORDER,
    TRADITIONAL_METHOD_ORDER,
    ordered_method_tokens,
    ordered_result_keys,
)
from report.method_titles import report_method_title
from report.runtime_metadata import infer_record_count


METHOD_CATALOG = {
    "shap": {
        "name": report_method_title("shap"),
        "short_name": "SHAP",
        "executive_item": "SHAP: Feature Attribution",
        "layer": "Explainability",
        "evidence_type": "Feature attribution",
        "articles": "Art. 13",
        "result_key": "explainability",
    },
    "lime": {
        "name": report_method_title("lime"),
        "short_name": "LIME",
        "executive_item": "LIME: Local Explanation",
        "layer": "Explainability",
        "evidence_type": "Local surrogate explanation",
        "articles": "Art. 13",
        "result_key": "lime",
    },
    "dice": {
        "name": report_method_title("dice"),
        "short_name": "DiCE",
        "executive_item": "DiCE: Counterfactual Explanation",
        "layer": "Explainability",
        "evidence_type": "Counterfactual explanation",
        "articles": "Art. 13",
        "result_key": "counterfactual",
    },
    "fairness": {
        "name": report_method_title("fairness"),
        "short_name": "Fairlearn",
        "executive_item": "Fairlearn",
        "layer": "Fairness",
        "evidence_type": "Group fairness metrics",
        "articles": "Art. 10",
        "result_key": "fairness",
    },
    "uncertainty": {
        "name": report_method_title("uncertainty"),
        "short_name": "MAPIE",
        "executive_item": "MAPIE",
        "layer": "Uncertainty",
        "evidence_type": "Conformal prediction",
        "articles": "Art. 14 / Art. 15",
        "result_key": "uncertainty",
    },
    "drift": {
        "name": report_method_title("drift"),
        "short_name": "Evidently",
        "executive_item": "Evidently + Feature Drift Tests",
        "layer": "Drift",
        "evidence_type": "Dataset / feature drift",
        "articles": "Art. 15 / Art. 61",
        "result_key": "drift",
    },
    LLM_GROUNDING: {
        "name": report_method_title(LLM_GROUNDING),
        "short_name": report_method_title(LLM_GROUNDING),
        "executive_item": report_method_title(LLM_GROUNDING),
        "layer": "LLM Grounding",
        "evidence_type": "Grounding evaluation",
        "articles": "Art. 13",
        "result_key": LLM_GROUNDING,
    },
    LLM_SELF_CONSISTENCY: {
        "name": report_method_title(LLM_SELF_CONSISTENCY),
        "short_name": report_method_title(LLM_SELF_CONSISTENCY),
        "executive_item": report_method_title(LLM_SELF_CONSISTENCY),
        "layer": "LLM Uncertainty",
        "evidence_type": "Self-consistency evaluation",
        "articles": "Art. 15",
        "result_key": LLM_SELF_CONSISTENCY,
    },
    LLM_SEMANTIC_DRIFT: {
        "name": report_method_title(LLM_SEMANTIC_DRIFT),
        "short_name": report_method_title(LLM_SEMANTIC_DRIFT),
        "executive_item": report_method_title(LLM_SEMANTIC_DRIFT),
        "layer": "LLM Drift",
        "evidence_type": "Semantic drift evaluation",
        "articles": "Art. 61",
        "result_key": LLM_SEMANTIC_DRIFT,
    },
    LLM_PROMPT_FAIRNESS: {
        "name": report_method_title(LLM_PROMPT_FAIRNESS),
        "short_name": report_method_title(LLM_PROMPT_FAIRNESS),
        "executive_item": report_method_title(LLM_PROMPT_FAIRNESS),
        "layer": "LLM Fairness",
        "evidence_type": "Differential prompt fairness",
        "articles": "Art. 10",
        "result_key": LLM_PROMPT_FAIRNESS,
    },
}

RESULT_KEY_TO_TOKEN = {
    details["result_key"]: token for token, details in METHOD_CATALOG.items()
}


def _value(context, name: str, default=None):
    if context is None:
        return default
    if isinstance(context, dict):
        return context.get(name, default)
    return getattr(context, name, default)


def _display(value, fallback="Not provided"):
    if value is None or value == "" or value == [] or value == {}:
        return fallback
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value) or fallback
    return value


def _natural_join(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])} and {values[-1]}"


def _method_groups(tokens: list[str]) -> list[dict]:
    """Group traditional methods by evidence layer for executive display."""
    if any(token in LLM_METHOD_ORDER for token in tokens):
        return [{
            "label": "LLM / Agentic Evidence",
            "items": [METHOD_CATALOG[token]["executive_item"] for token in tokens],
        }]

    layer_groups = (
        ("Explainability Evidence", ("shap", "lime", "dice")),
        ("Fairness Evidence", ("fairness",)),
        ("Uncertainty Evidence", ("uncertainty",)),
        ("Drift Evidence", ("drift",)),
    )
    groups = []
    for label, layer_tokens in layer_groups:
        items = [
            METHOD_CATALOG[token]["executive_item"]
            for token in layer_tokens
            if token in tokens
        ]
        if items:
            groups.append({"label": label, "items": items})
    return groups


def _grouped_method_fact(label: str, tokens: list[str]) -> dict:
    return {
        "label": label,
        "value": "None" if not tokens else None,
        "groups": _method_groups(tokens),
    }


def _executive_method_summary(tokens: list[str]) -> str:
    if not tokens:
        return "no evidence methods"
    if any(token in LLM_METHOD_ORDER for token in tokens):
        return _natural_join([METHOD_CATALOG[token]["name"] for token in tokens])

    summaries = []
    for group in _method_groups(tokens):
        method_names = [item.split(":", 1)[0] for item in group["items"]]
        summaries.append(f"{group['label']} — {_natural_join(method_names)}")
    return "; ".join(summaries)


def _clean_label(value, fallback="unspecified") -> str:
    """Format internal enum-style values for report prose only."""
    if value is None or value == "":
        return fallback
    text = str(value).strip().lower()
    special = {
        "llm": "LLM",
        "llm_generation": "LLM generation",
        "time_series": "time-series",
        "binary_classification": "binary classification",
        "multiclass_classification": "multiclass classification",
        "anomaly_detection": "anomaly detection",
        "partially_compliant": "partially compliant",
    }
    return special.get(text, text.replace("_", " "))


def _executive_system_description(audit_context) -> str:
    if _is_llm(audit_context):
        system_type = str(_value(audit_context, "system_type", "") or "").lower()
        return "agentic LLM system" if system_type == "agentic" else "LLM system"

    modality = _clean_label(_value(audit_context, "modality"), "traditional ML")
    task = _clean_label(_value(audit_context, "task_type"), "model evaluation")
    return f"{modality} {task} system"


def _web_path(value):
    if value in (None, ""):
        return None
    return str(value).replace("\\", "/")


def _layer_display(value):
    labels = {
        "explainability": "Explainability",
        "fairness": "Fairness",
        "uncertainty": "Uncertainty",
        "drift": "Drift",
        "counterfactual": "Explainability",
        "llm_explainability": "LLM Grounding",
        "llm_uncertainty": "LLM Uncertainty",
        "llm_drift": "LLM Drift",
        "llm_fairness": "LLM Fairness",
    }
    text = str(value or "Evidence")
    return labels.get(text, text.replace("_", " ").title())


def _is_llm(audit_context) -> bool:
    system_type = str(_value(audit_context, "system_type", "") or "").lower()
    task_type = str(_value(audit_context, "task_type", "") or "").lower()
    return system_type in {"llm", "agentic"} or task_type == "llm_generation"


def _trace(results: dict) -> dict:
    value = results.get("_cbep_trace", {})
    return value if isinstance(value, dict) else {}


def _result_status(result: Any) -> str:
    if not isinstance(result, dict):
        return "not_available"
    explicit = str(result.get("status", "") or "").strip().lower()
    if explicit:
        return explicit
    if result.get("error"):
        return "failed"
    return "completed"


def format_status_badge(status: str) -> dict[str, str]:
    """Return a normalized status label and CSS class."""
    normalized = str(status or "not_available").strip().lower()
    labels = {
        "completed": "Completed",
        "skipped": "Skipped",
        "failed": "Failed",
        "not_applicable": "Not applicable",
        "not_selected": "Not selected",
        "not_executed": "Not executed",
        "not_available": "Not available",
        "partial": "Partial coverage",
        "limited": "Execution limited",
        "metadata_only": "Metadata-only",
    }
    return {
        "status": normalized,
        "label": labels.get(normalized, normalized.replace("_", " ").title()),
        "css_class": f"status-{normalized.replace('_', '-')}",
    }


def _status_display(status) -> str:
    if not status:
        return "Not provided"
    return format_status_badge(str(status))["label"]


def _candidate_tokens(results: dict, audit_context) -> list[str]:
    return list(LLM_METHOD_ORDER if _is_llm(audit_context) else TRADITIONAL_METHOD_ORDER)


def _selected_tokens(results: dict) -> list[str]:
    final_plan = _trace(results).get("final_plan", [])
    return ordered_method_tokens(final_plan) if isinstance(final_plan, list) else []


def _incompatible_tokens(results: dict) -> list[str]:
    assessment = _trace(results).get("task_compatibility_assessment", {})
    values = assessment.get("incompatible_methods", []) if isinstance(assessment, dict) else []
    return list(values) if isinstance(values, list) else []


def _method_result(results: dict, token: str):
    details = METHOD_CATALOG.get(token, {})
    return results.get(details.get("result_key", token))


def _method_status(results: dict, token: str) -> str:
    selected = token in _selected_tokens(results)
    incompatible = token in _incompatible_tokens(results)
    result = _method_result(results, token)
    if result is not None:
        return _result_status(result)
    if incompatible:
        return "not_applicable"
    if selected:
        return "not_executed"
    return "not_selected"


def _compatibility_reason(token: str, task_type: str) -> str:
    task = task_type or "the declared task"
    if task == "forecasting":
        reasons = {
            "fairness": "Forecasting is not a group classification task in the current implementation.",
            "dice": "The current counterfactual runner targets classification or regression estimators, not this forecasting wrapper.",
            "lime": "The current LIME pathway is not enabled for forecasting.",
            "uncertainty": "The current MAPIE runner does not support the forecasting wrapper.",
        }
        return reasons.get(token, f"The method is not compatible with {task}.")
    if task == "anomaly_detection":
        reasons = {
            "fairness": "The current fairness runner supports classification group metrics, not anomaly scores.",
            "dice": "IsolationForest-style anomaly detection does not expose the standard class probabilities required by the current DiCE runner.",
            "lime": "The current LIME pathway is not enabled for anomaly detection.",
            "uncertainty": "The current MAPIE runner is not compatible with IsolationForest-style anomaly detection.",
        }
        return reasons.get(token, f"The method is not compatible with {task}.")
    return f"The method is not compatible with task type {task}."


def build_executive_summary(results, audit_context, governance_context, provider_name):
    selected = _selected_tokens(results)
    statuses = [_method_status(results, item) for item in selected]
    completed_tokens = [
        item for item in selected
        if item in METHOD_CATALOG and _method_status(results, item) == "completed"
    ]
    skipped_tokens = [
        item for item in selected
        if item in METHOD_CATALOG and _method_status(results, item) != "completed"
    ]

    if statuses and all(status == "completed" for status in statuses):
        overall = "completed"
    elif completed_tokens:
        overall = "partial"
    elif skipped_tokens:
        overall = "limited"
    else:
        overall = "not_available"

    risk = _clean_label(_value(audit_context, "risk_tier"), "unspecified")
    domain = _clean_label(_value(audit_context, "application_domain"), "unspecified")
    system_description = _executive_system_description(audit_context)
    method_text = _executive_method_summary(selected)

    paragraphs = [
        (
            f"This audit report evaluates {provider_name}, a {risk}-risk "
            f"{system_description} in the {domain} domain. CBEP selected "
            f"{method_text} using the "
            "S4 governance context, S5 audit context, EU AI Act article mapping, "
            "governance priorities, and task compatibility rules."
        )
    ]

    shap_result = results.get("explainability", {})
    shap_metrics = shap_result.get("metrics", {}) if isinstance(shap_result, dict) else {}
    top_features = shap_metrics.get("top_features", []) if isinstance(shap_metrics, dict) else []
    if top_features:
        paragraphs.append(
            "The explainability evidence identifies "
            f"{', '.join(map(str, top_features[:4]))} as the leading model features "
            "in the evaluated sample."
        )
    fairness_result = results.get("fairness")
    if isinstance(fairness_result, dict) and _result_status(fairness_result) == "completed":
        sensitive = _value(audit_context, "sensitive_feature_columns", []) or []
        paragraphs.append(
            "Fairness evidence was generated for "
            f"{', '.join(map(str, sensitive)) or 'the configured sensitive features'}."
        )
    if _is_llm(audit_context) and skipped_tokens:
        paragraphs.append(
            "The LLM / Agentic resource contract and golden-set inputs were validated, "
            "but execution-level evidence remains limited by model loadability."
        )

    return {
        "paragraphs": paragraphs,
        "facts": [
            {"label": "Audit case", "value": provider_name},
            {"label": "Risk tier", "value": _display(_value(audit_context, "risk_tier"))},
            {"label": "Application domain", "value": _display(_value(audit_context, "application_domain"))},
            {"label": "System / task", "value": f"{_display(_value(audit_context, 'system_type'))} / {_display(_value(audit_context, 'task_type'))}"},
            {"label": "Governance score", "value": _display(_value(governance_context, "governance_score"))},
            {"label": "Governance verdict", "value": _display(_value(governance_context, "governance_verdict"))},
            _grouped_method_fact("Selected methods", selected),
            _grouped_method_fact("Completed methods", completed_tokens),
            _grouped_method_fact("Skipped / unavailable", skipped_tokens),
        ],
        "selected_methods": [METHOD_CATALOG[token]["short_name"] for token in selected],
        "overall_status": format_status_badge(overall),
    }


def build_audit_scope(audit_context, governance_context, resource_context=None):
    common_rows = [
        {"label": "S4 governance score", "value": _display(_value(governance_context, "governance_score"))},
        {"label": "S4 governance verdict", "value": _display(_value(governance_context, "governance_verdict"))},
        {"label": "S5 risk tier", "value": _display(_value(audit_context, "risk_tier"))},
        {"label": "Application domain", "value": _display(_value(audit_context, "application_domain"))},
        {"label": "System type / modality", "value": f"{_display(_value(audit_context, 'system_type'))} / {_display(_value(audit_context, 'modality'))}"},
        {"label": "Task type", "value": _display(_value(audit_context, "task_type"))},
        {"label": "CSP satisfied", "value": _display(_value(audit_context, "csp_satisfied"))},
    ]
    contract_rows = [
        {"label": "Target column", "value": _display(_value(audit_context, "target_column"))},
        {"label": "Positive label", "value": _display(_value(audit_context, "positive_label"))},
        {"label": "Sensitive features", "value": _display(_value(audit_context, "sensitive_feature_columns", []), "None configured")},
    ]
    domain_scores = _value(governance_context, "domain_scores", {}) or {}
    return {
        "common_rows": common_rows,
        "contract_rows": contract_rows,
        "domain_scores": [
            {"domain": name, "score": score} for name, score in domain_scores.items()
        ],
    }


def build_resource_summary(audit_context, resource_context):
    context = resource_context or {}
    status = context.get("model_status") or "loaded"
    artifact_kind = context.get("artifact_kind")
    if not artifact_kind:
        if status == "metadata_only":
            artifact_kind = "Metadata-only LLM model folder"
        elif _value(audit_context, "model_format") == "model_directory":
            artifact_kind = "Model directory"
        else:
            artifact_kind = "Single model file"

    warnings = list(context.get("load_warnings") or [])
    if status == "metadata_only":
        warnings.append(
            "The LLM model directory contains configuration metadata but no loadable model weights."
        )

    model_container = context.get("model_container")
    narrative = f"The S5 model reference was resolved as a {artifact_kind.lower()}."
    if model_container and status != "metadata_only":
        narrative += f" The executable entrypoint was loaded as {model_container}."
    if status == "metadata_only":
        narrative = (
            "The LLM / Agentic resource contract was validated, and the golden "
            "evaluation set and model metadata were loaded. However, execution-level "
            "LLM evidence was skipped because the provided model artifact contains "
            "configuration metadata but no loadable model weights."
        )

    is_llm = _is_llm(audit_context)
    contract_name = (
        "LLM / Agentic golden-set contract"
        if is_llm
        else "Traditional ML dataset contract"
    )
    rows = [
        {"label": "Artifact type", "value": artifact_kind},
        {"label": "Artifact URI", "value": _display(_value(audit_context, "model_artifact_uri"))},
        {"label": "Resolved model path", "value": _display(_web_path(context.get("resolved_path")))},
        {"label": "Model format", "value": _display(_value(audit_context, "model_format"))},
        {"label": "Model framework", "value": _display(_value(audit_context, "model_framework"))},
        {"label": "Model type", "value": _display(_value(audit_context, "model_type"))},
        {"label": "Model entrypoint", "value": _display(_value(audit_context, "model_entrypoint"))},
        {"label": "Loaded model form", "value": _display(model_container)},
        {"label": "Model loadability", "value": "Loadable" if context.get("model_is_loadable", True) else "Not loadable"},
        {"label": "Model artifact status", "value": _status_display(status)},
        {"label": "Loaded metadata files", "value": _display(context.get("loaded_metadata_files", []), "None")},
    ]

    if is_llm:
        golden_count = context.get("golden_set_records")
        if golden_count is None:
            golden_count = infer_record_count(context.get("golden_set"))
        rows.extend([
            {"label": "Golden set URI", "value": _display(_value(audit_context, "golden_set_uri"))},
            {"label": "Golden-set records", "value": _display(golden_count, "Unavailable")},
            {"label": "System prompt URI", "value": _display(_value(audit_context, "system_prompt_uri"))},
            {"label": "RAG manifest URI", "value": _display(_value(audit_context, "rag_manifest_uri"))},
            {"label": "Guardrail config URI", "value": _display(_value(audit_context, "guardrail_config_uri"))},
            {"label": "Golden set loaded", "value": _display(context.get("golden_set_loaded"))},
            {"label": "Metadata-only", "value": "Yes" if status == "metadata_only" else "No"},
            {"label": "Training dataset required", "value": "No"},
            {"label": "Evaluation dataset required", "value": "No"},
        ])
    else:
        rows.extend([
            {"label": "Training dataset URI", "value": _display(_value(audit_context, "training_dataset_uri"))},
            {"label": "Evaluation dataset URI", "value": _display(_value(audit_context, "evaluation_dataset_uri"))},
            {"label": "Training dataset loaded", "value": _display(context.get("training_dataset_loaded"))},
            {"label": "Evaluation dataset loaded", "value": _display(context.get("evaluation_dataset_loaded"))},
            {"label": "Training rows", "value": _display(context.get("training_dataset_rows"), "Unavailable")},
            {"label": "Evaluation rows", "value": _display(context.get("evaluation_dataset_rows"), "Unavailable")},
        ])

    return {
        "narrative": narrative,
        "artifact_kind": artifact_kind,
        "status": format_status_badge("metadata_only" if status == "metadata_only" else "completed"),
        "contract_name": contract_name,
        "rows": rows,
        "warnings": list(dict.fromkeys(warnings)),
    }


def build_cbep_decision_table(results, audit_context):
    trace = _trace(results)
    selected = _selected_tokens(results)
    incompatible = _incompatible_tokens(results)
    task_type = str(_value(audit_context, "task_type", "") or "")
    rows = []
    for token in _candidate_tokens(results, audit_context):
        info = METHOD_CATALOG[token]
        if token in selected:
            reason = f"Selected by CBEP and compatible with {task_type}."
        elif token in incompatible:
            reason = _compatibility_reason(token, task_type)
        else:
            reason = "Not required by the minimum sufficient evidence plan."
        rows.append({
            "method": info["short_name"],
            "layer": info["layer"],
            "articles": info["articles"],
            "selected": "Yes" if token in selected else "No",
            "reason": reason,
        })
    return {
        "rows": rows,
        "base_plan": [
            METHOD_CATALOG.get(item, {"short_name": item})["short_name"]
            for item in ordered_method_tokens(trace.get("base_plan", []))
        ],
        "article_plan": [
            METHOD_CATALOG.get(item, {"short_name": item})["short_name"]
            for item in ordered_method_tokens(trace.get("article_plan", []))
        ],
        "governance_adjustments": _value(trace.get("governance_context", {}), "adjustments", []) or [],
        "csp_adjustments": trace.get("csp_adjustments", []) or [],
    }


def _main_output(token: str, result: Any, status: str) -> str:
    if status == "not_applicable":
        return "No execution result"
    if not isinstance(result, dict):
        return "No structured output"
    if status in {"skipped", "failed"}:
        return "Execution skipped"
    metrics = result.get("metrics", {})
    if token == "shap":
        return f"{len(metrics.get('top_features', []))} ranked features"
    if token == "lime":
        return f"{metrics.get('features_explained', 0)} local feature contributions"
    if token == "fairness":
        count = len(metrics.get("sensitive_features", []))
        return f"Metrics for {count} sensitive features"
    if token == "uncertainty":
        return f"Coverage {metrics.get('coverage', 'not reported')}"
    if token == "drift":
        decision = "detected" if metrics.get("dataset_drift_detected") else "not detected"
        return (
            f"Dataset drift {decision}; share "
            f"{metrics.get('drift_share', 'not reported')}"
        )
    if token == "dice":
        return (
            f"{metrics.get('counterfactuals_count', 0)} counterfactual(s); "
            f"{metrics.get('changed_features_count', 0)} changed feature(s) "
            "in the first result"
        )
    if token.startswith("llm_"):
        return "Structured LLM evidence result"
    return "Structured evidence result"


def _result_limitation(token: str, result: Any, status: str, audit_context) -> str:
    if isinstance(result, dict):
        limitations = result.get("limitations", [])
        if limitations:
            return str(limitations[0])
        if status in {"skipped", "failed", "not_applicable"}:
            return str(result.get("summary") or "No evidence was produced.")
    return "None identified in this evidence result."


def build_evidence_coverage_matrix(results, audit_context):
    rows = []
    for token in _candidate_tokens(results, audit_context):
        info = METHOD_CATALOG[token]
        result = _method_result(results, token)
        if not isinstance(result, dict):
            continue
        status = _method_status(results, token)
        rows.append({
            "layer": _layer_display(result.get("layer", info["layer"])),
            "evidence_type": info["evidence_type"],
            "method": info["short_name"],
            "status": format_status_badge(status),
            "articles": ", ".join(result.get("article_mapping", [])),
            "main_output": _main_output(token, result, status),
            "limitation": _result_limitation(token, result, status, audit_context),
        })
    return rows


def _fairness_feature_results(result: dict) -> list[tuple[str, dict]]:
    metrics = result.get("metrics", {})
    per_feature = metrics.get("per_feature", {}) if isinstance(metrics, dict) else {}
    return list(per_feature.items()) if isinstance(per_feature, dict) else []


def build_evidence_narrative(token: str, result: dict, audit_context) -> dict:
    info = METHOD_CATALOG[token]
    status = _result_status(result)
    evidence_metrics = result.get("metrics", {})
    artifacts = list(result.get("artifacts") or [])
    metric_cards = []
    table = None

    if token == "shap":
        ranked = evidence_metrics.get("feature_importance", [])
        table = {
            "headers": ["Rank", "Feature", "Mean absolute importance"],
            "rows": [
                [index, item.get("feature"), item.get("importance")]
                for index, item in enumerate(ranked[:10], 1)
            ],
        }
    elif token == "fairness":
        feature_results = _fairness_feature_results(result)
        table = {
            "headers": [
                "Sensitive feature",
                "Demographic parity difference",
                "Equalized odds difference",
            ],
            "rows": [
                [
                    name,
                    values.get("demographic_parity_difference", "Unavailable"),
                    values.get("equalized_odds_difference", "Unavailable"),
                ]
                for name, values in feature_results
            ],
        }
    elif token == "lime":
        table = {
            "headers": ["Feature condition", "Local weight"],
            "rows": [
                [item.get("feature_condition"), item.get("weight")]
                for item in evidence_metrics.get("feature_contributions", [])
            ],
        }
    elif token == "dice":
        table = {
            "headers": [
                "Feature",
                "Original value",
                "Counterfactual value",
                "Delta",
            ],
            "rows": [
                [
                    (
                        f"{item.get('feature')} (sensitive / immutable)"
                        if item.get("is_sensitive_or_immutable")
                        else item.get("feature")
                    ),
                    item.get("original_value"),
                    item.get("counterfactual_value"),
                    item.get("delta") if item.get("delta") is not None else "N/A",
                ]
                for item in evidence_metrics.get("changed_features", [])
            ],
        }
    elif token == "drift":
        table = {
            "headers": [
                "Rank",
                "Feature",
                "Type",
                "Method",
                "Statistic",
                "p-value",
                "Drift Detected",
            ],
            "rows": [
                [
                    index,
                    item.get("feature"),
                    item.get("feature_type"),
                    item.get("method"),
                    item.get("statistic"),
                    item.get("p_value") if item.get("p_value") is not None else "N/A",
                    "Yes" if item.get("drift_detected") else "No",
                ]
                for index, item in enumerate(
                    evidence_metrics.get("top_drifted_columns", []), 1
                )
            ],
        }

    for key, value in evidence_metrics.items():
        if key == "runtime_seconds" or isinstance(value, (dict, list, tuple)):
            continue
        metric_cards.append(
            {"label": key.replace("_", " ").title(), "value": value}
        )

    plot = next(
        (_web_path(path) for path in artifacts if str(path).lower().endswith((".png", ".jpg", ".jpeg", ".svg"))),
        None,
    )
    output = next(
        (_web_path(path) for path in artifacts if _web_path(path) != plot),
        None,
    )

    return {
        "key": info["result_key"],
        "evidence_id": result.get("evidence_id"),
        "method": info["name"],
        "layer": _layer_display(result.get("layer", info["layer"])),
        "evidence_type": info["evidence_type"],
        "articles": ", ".join(result.get("article_mapping", [])),
        "status": format_status_badge(status),
        "narrative": result.get("summary") or "No evidence summary was provided.",
        "key_findings": list(result.get("key_findings") or []),
        "metrics": metric_cards,
        "table": table,
        "limitations": list(result.get("limitations") or []),
        "plot": plot,
        "output": output,
        "output_label": (
            "Open full structured DiCE JSON artifact"
            if token == "dice"
            else f"Open generated {info['name']} artifact"
        ),
    }


def build_evidence_findings(results, audit_context):
    findings = []
    for token in _candidate_tokens(results, audit_context):
        result = _method_result(results, token)
        if isinstance(result, dict) and _result_status(result) != "not_applicable":
            findings.append(build_evidence_narrative(token, result, audit_context))
    return findings


def build_method_applicability(results, audit_context):
    rows = []
    selected = _selected_tokens(results)
    for token in _candidate_tokens(results, audit_context):
        result = _method_result(results, token)
        if not isinstance(result, dict):
            continue
        status = _method_status(results, token)
        if status == "completed":
            continue
        info = METHOD_CATALOG[token]
        rows.append({
            "method": info["name"],
            "status": format_status_badge(status),
            "reason": _result_limitation(token, result, status, audit_context),
            "selected": "Yes" if token in selected else "No",
        })

    if _is_llm(audit_context):
        context_note = (
            "Traditional SHAP, Fairlearn, MAPIE, DiCE, and tabular Evidently methods "
            "are not applied directly. CBEP selected the dedicated LLM evidence pathway."
        )
    else:
        context_note = (
            "CBEP creates an article- and risk-driven plan and then applies task "
            "compatibility screening. Incompatible methods are excluded to avoid "
            "misleading or invalid evidence."
        )
    return {"intro": context_note, "rows": rows}


def build_runtime_reproducibility(
    results,
    audit_context,
    runtime_context,
):
    """Build a report-only view of execution and environment metadata."""
    runtime_context = runtime_context or {}
    selected = _selected_tokens(results)
    candidate_tokens = _candidate_tokens(results, audit_context)

    runtime_rows = []
    for token in candidate_tokens:
        result = _method_result(results, token)
        status = _method_status(results, token)
        if token not in selected and result is None and status != "not_applicable":
            continue
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        runtime_seconds = metrics.get("runtime_seconds", 0.0) if isinstance(metrics, dict) else 0.0
        try:
            runtime_display = f"{float(runtime_seconds):.3f}"
        except (TypeError, ValueError):
            runtime_display = "Unavailable"
        runtime_rows.append(
            {
                "method": METHOD_CATALOG[token]["name"],
                "status": format_status_badge(status),
                "runtime_seconds": runtime_display,
            }
        )

    artifact_rows = []
    seen_artifacts = set()

    def add_artifact(artifact_type, method, path):
        if not path:
            return
        normalized_path = str(path)
        if normalized_path in seen_artifacts:
            return
        seen_artifacts.add(normalized_path)
        artifact_rows.append(
            {"type": artifact_type, "method": method, "path": normalized_path}
        )

    add_artifact(
        "HTML audit report",
        "Report exporter",
        runtime_context.get("html_report_path"),
    )
    if runtime_context.get("pdf_generated"):
        add_artifact(
            "PDF audit report",
            "Report exporter",
            runtime_context.get("pdf_report_path"),
        )

    for result_key in ordered_result_keys(results):
        result = results[result_key]
        if not isinstance(result, dict):
            continue
        token = RESULT_KEY_TO_TOKEN.get(result_key)
        title = METHOD_CATALOG[token]["name"] if token in METHOD_CATALOG else _display(result.get("method"), result_key)
        for artifact_path in result.get("artifacts") or []:
            suffix = str(artifact_path).rsplit(".", 1)[-1].upper() if "." in str(artifact_path) else "Artifact"
            add_artifact(f"{suffix} evidence artifact", title, artifact_path)

    total_runtime = runtime_context.get("total_runtime_seconds")
    try:
        total_runtime_display = f"{float(total_runtime):.3f} seconds"
    except (TypeError, ValueError):
        total_runtime_display = "Unavailable"

    common_facts = [
        {"label": "Run timestamp", "value": _display(runtime_context.get("run_timestamp"))},
        {"label": "Python version", "value": _display(runtime_context.get("python_version"))},
        {"label": "Platform", "value": _display(runtime_context.get("platform"))},
        {"label": "Total runtime", "value": total_runtime_display},
        {"label": "HTML report path", "value": _display(runtime_context.get("html_report_path"))},
        {"label": "PDF report path", "value": _display(runtime_context.get("pdf_report_path") if runtime_context.get("pdf_generated") else None, "Not generated")},
    ]

    return {
        "common_facts": common_facts,
        "runtime_rows": runtime_rows,
        "artifact_rows": artifact_rows,
        "measurement_scope": _display(runtime_context.get("measurement_scope")),
    }


def build_report_model(
    results: dict,
    audit_context,
    governance_context,
    resource_context,
    provider_name: str,
    runtime_context=None,
) -> dict:
    resource_summary = build_resource_summary(audit_context, resource_context)
    applicability = build_method_applicability(results, audit_context)
    return {
        "executive_summary": build_executive_summary(
            results, audit_context, governance_context, provider_name
        ),
        "audit_scope": build_audit_scope(
            audit_context, governance_context, resource_context
        ),
        "resource_summary": resource_summary,
        "runtime_reproducibility": build_runtime_reproducibility(
            results,
            audit_context,
            runtime_context,
        ),
        "cbep_summary": build_cbep_decision_table(results, audit_context),
        "coverage_matrix": build_evidence_coverage_matrix(results, audit_context),
        "evidence_findings": build_evidence_findings(results, audit_context),
        "applicability": applicability,
    }
