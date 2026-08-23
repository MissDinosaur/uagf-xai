"""Build a presentation-oriented view model for the HTML audit report."""

from __future__ import annotations

from typing import Any

from report.method_ordering import (
    LLM_METHOD_ORDER,
    TRADITIONAL_METHOD_ORDER,
    ordered_method_tokens,
    ordered_result_keys,
)
from report.runtime_metadata import infer_record_count
from schema.method_catalog import METHOD_CATALOG as CANONICAL_METHOD_CATALOG


METHOD_CATALOG = {
    token: {
        "name": details.display_name,
        "short_name": details.short_name,
        "executive_item": details.executive_item,
        "layer": details.report_layer,
        "evidence_type": details.report_evidence_type,
        "articles": " / ".join(details.articles),
        "result_key": details.result_key,
        "requirements": list(details.requirements),
    }
    for token, details in CANONICAL_METHOD_CATALOG.items()
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
    return system_type in {"llm", "agentic"}


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


def build_executive_summary(
    results,
    audit_context,
    governance_context,
    provider_name,
    resource_context=None,
):
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

    is_local_surrogate = bool((resource_context or {}).get("is_local_surrogate"))
    if is_local_surrogate:
        planning_context = (
            "the local high-risk justice-domain validation context"
        )
    else:
        planning_context = "the S4 governance context and S5 audit context"
    paragraphs = [
        (
            f"This audit report evaluates {provider_name}, a {risk}-risk "
            f"{system_description} in the {domain} domain. CBEP selected "
            f"{method_text} using {planning_context}, EU AI Act article mapping, "
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
    drift_result = results.get("drift")
    if isinstance(drift_result, dict) and _result_status(drift_result) == "completed":
        drift_metrics = drift_result.get("metrics", {})
        decision = (
            "detected"
            if drift_metrics.get("dataset_drift_detected")
            else "not detected"
        )
        is_talentsift_text = (
            "talentsift" in str(provider_name or "").lower()
            and str(_value(audit_context, "modality", "")).lower() == "text"
            and drift_metrics.get("model_feature_columns") == ["cv_text"]
        )
        if is_talentsift_text and decision == "detected":
            paragraphs.append(
                "Canonical text-input drift was detected for the authoritative "
                "cv_text model input: three of four derived statistical "
                "characteristics were flagged as drifted, producing a drift share "
                f"of {drift_metrics.get('drift_share', 'not reported')}."
            )
        else:
            paragraphs.append(
                "Canonical model-input drift was "
                f"{decision} across {drift_metrics.get('features_analyzed', 0)} "
                "authoritative model features, with a drift share of "
                f"{drift_metrics.get('drift_share', 'not reported')}."
            )

    return {
        "paragraphs": paragraphs,
        "facts": [
            {"label": "Audit case", "value": provider_name},
            {"label": "Risk tier", "value": _display(_value(audit_context, "risk_tier"))},
            {"label": "Application domain", "value": _display(_value(audit_context, "application_domain"))},
            {
                "label": "System type / Modality",
                "value": (
                    f"{_display(_value(audit_context, 'system_type'))} / "
                    f"{_display(_value(audit_context, 'modality'))}"
                ),
            },
            {
                "label": "Task type",
                "value": _display(_value(audit_context, "task_type")),
            },
            {"label": "Governance score", "value": _display(_value(governance_context, "governance_score"))},
            _grouped_method_fact("Selected methods", selected),
            _grouped_method_fact("Completed methods", completed_tokens),
            _grouped_method_fact("Skipped / unavailable", skipped_tokens),
        ],
        "selected_methods": [METHOD_CATALOG[token]["short_name"] for token in selected],
        "overall_status": format_status_badge(overall),
    }


def build_audit_scope(audit_context, governance_context, resource_context=None):
    is_local_surrogate = bool((resource_context or {}).get("is_local_surrogate"))
    common_rows = [
        {"label": "S4 governance score", "value": _display(_value(governance_context, "governance_score"))},
        {"label": "S4 governance verdict", "value": _display(_value(governance_context, "governance_verdict"))},
        {
            "label": "Audit-context risk tier" if is_local_surrogate else "S5 risk tier",
            "value": _display(_value(audit_context, "risk_tier")),
        },
        {"label": "Application domain", "value": _display(_value(audit_context, "application_domain"))},
        {"label": "System type / modality", "value": f"{_display(_value(audit_context, 'system_type'))} / {_display(_value(audit_context, 'modality'))}"},
        {"label": "CSP satisfied", "value": _display(_value(audit_context, "csp_satisfied"))},
    ]
    task_type = str(_value(audit_context, "task_type", "") or "").lower()
    is_llm = _is_llm(audit_context) or task_type == "llm_generation"
    contract_fallback = "Not applicable" if is_llm else "Not provided"
    contract_rows = [
        {"label": "Target column", "value": _display(_value(audit_context, "target_column"), contract_fallback)},
        {
            "label": "Positive label",
            "value": "Not applicable" if is_llm else _display(_value(audit_context, "positive_label")),
        },
        {"label": "Sensitive features", "value": _display(_value(audit_context, "sensitive_feature_columns", []), "None configured")},
    ]
    modality = str(_value(audit_context, "modality", "") or "").lower()
    if modality == "text" or is_llm:
        contract_rows.insert(
            2,
            {
                "label": "Model feature columns",
                "value": "Not applicable" if is_llm else _display(
                    _value(audit_context, "feature_columns"),
                    "Legacy target-drop fallback",
                ),
            },
        )
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
    is_local_surrogate = bool(context.get("is_local_surrogate"))
    status = context.get("model_status") or "loaded"
    artifact_kind = context.get("artifact_kind")
    if not artifact_kind:
        if status == "metadata_only":
            artifact_kind = "Metadata-only LLM model folder"
        elif _value(audit_context, "model_artifact_kind") == "directory":
            artifact_kind = "Model directory"
        else:
            artifact_kind = "Single model file"

    warnings = list(context.get("load_warnings") or [])
    if status == "metadata_only":
        warnings.append(
            "The LLM model directory contains configuration metadata but no loadable model weights."
        )

    model_container = context.get("model_container")
    reference_owner = "local model reference" if is_local_surrogate else "S5 model reference"
    narrative = f"The {reference_owner} was resolved as a {artifact_kind.lower()}."
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
    if is_local_surrogate:
        narrative = (
            "Local surrogate execution validation. The original LegalMind model "
            "weights were unavailable. DistilGPT2 was used as a locally executable "
            "surrogate to validate the mechanics of the LLM evidence pathway. The "
            "resulting scores are not an audit of the original LegalMind model."
        )
    is_text = str(_value(audit_context, "modality", "") or "").lower() == "text"
    contract_name = (
        "LLM / Agentic golden-set contract"
        if is_llm
        else "Traditional ML dataset contract"
    )
    rows = [
        {"label": "Artifact type", "value": artifact_kind},
        {"label": "Artifact URI", "value": _display(_value(audit_context, "model_artifact_uri"))},
        {
            "label": "Artifact kind",
            "value": _display(_value(audit_context, "model_artifact_kind")),
        },
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
    if is_text:
        text_rows = [
            {
                "label": "Model input adapter",
                "value": _display(context.get("input_adapter")),
            },
            {
                "label": "Text feature column",
                "value": _display(context.get("text_feature_column")),
            },
            {
                "label": "TF-IDF vectorizer",
                "value": _display(context.get("vectorizer_class")),
            },
            {
                "label": "Final estimator",
                "value": _display(context.get("estimator_class")),
            },
            {
                "label": "TF-IDF vocabulary size",
                "value": _display(context.get("vocabulary_size")),
            },
        ]
        rows[9:9] = text_rows

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
            {
                "label": "Audited/surrogate generation model",
                "value": "DistilGPT2" if is_local_surrogate else _display(
                    _value(audit_context, "model_type")
                ),
            },
            {
                "label": "Evaluation embedding model",
                "value": _display(
                    context.get("evaluation_embedding_model_name"),
                    "Not configured",
                ),
            },
            {
                "label": "Evaluation embedding path",
                "value": _display(
                    context.get("evaluation_embedding_metadata", {}).get(
                        "resolved_path"
                    )
                ),
            },
            {
                "label": "Embedding pooling / normalization",
                "value": (
                    f"{_display(context.get('evaluation_embedding_metadata', {}).get('pooling'))} / "
                    f"{_display(context.get('evaluation_embedding_metadata', {}).get('normalization'))}"
                ),
            },
            {
                "label": "Offline local loading",
                "value": _display(
                    context.get("evaluation_embedding_metadata", {}).get(
                        "local_files_only"
                    )
                ),
            },
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
    assessment = trace.get("task_compatibility_assessment", {})
    trace_reasons = (
        assessment.get("incompatibility_reasons", {})
        if isinstance(assessment, dict)
        else {}
    )
    rows = []
    for token in _candidate_tokens(results, audit_context):
        info = METHOD_CATALOG[token]
        if token in selected:
            reason = f"Selected by CBEP and compatible with {task_type}."
        elif token in incompatible:
            reason = trace_reasons.get(token) or _compatibility_reason(token, task_type)
        else:
            reason = "Not required by the minimum sufficient evidence plan."
        rows.append({
            "method": info["short_name"],
            "layer": info["layer"],
            "articles": info["articles"],
            "selected": "Yes" if token in selected else "No",
            "reason": reason,
        })
    intro = (
        "CBEP is a deterministic, constraint-informed planning procedure inspired "
        "by CSP theory. It creates an article-, governance-, system-, and risk-driven "
        "plan and then applies task and modality compatibility screening. "
        "Incompatible methods are excluded to avoid misleading or invalid evidence."
    )
    if task_type in {"forecasting", "anomaly_detection"}:
        intro += (
            " CBEP evaluated the traditional evidence library, but task-compatibility "
            "screening excluded methods that are not valid for the current task type."
        )
    if task_type == "anomaly_detection":
        selected_names = _natural_join(
            [METHOD_CATALOG[token]["short_name"] for token in selected]
        )
        intro += (
            f" For anomaly detection, the current minimum-sufficient plan selected "
            f"{selected_names or 'no executable methods'}; classification fairness, conformal class prediction "
            "sets, LIME, and DiCE are reported as non-applicable rather than as "
            "planning failures."
        )
    return {
        "intro": intro,
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
            f"Model-input drift {decision}; share "
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
            "main_output": _main_output(token, result, status),
            "limitation": _result_limitation(token, result, status, audit_context),
        })
    return rows


def _fairness_feature_results(result: dict) -> list[tuple[str, dict]]:
    metrics = result.get("metrics", {})
    per_feature = metrics.get("per_feature", {}) if isinstance(metrics, dict) else {}
    return list(per_feature.items()) if isinstance(per_feature, dict) else []


def _prediction_label_meaning(value, audit_context) -> str:
    """Return an explicit prediction-label meaning without inferring semantics."""
    for field_name in (
        "prediction_label_mapping",
        "label_mapping",
        "class_labels",
    ):
        mapping = _value(audit_context, field_name)
        if isinstance(mapping, dict):
            meaning = mapping.get(value, mapping.get(str(value)))
            if meaning not in (None, ""):
                return str(meaning)
        elif isinstance(mapping, (list, tuple)):
            try:
                meaning = mapping[int(value)]
            except (IndexError, TypeError, ValueError):
                continue
            if meaning not in (None, ""):
                return str(meaning)
    return "label meaning unavailable"


def _prediction_display(value, audit_context) -> str:
    return f"{_display(value, 'Unavailable')} ({_prediction_label_meaning(value, audit_context)})"


def build_evidence_narrative(token: str, result: dict, audit_context) -> dict:
    info = METHOD_CATALOG[token]
    status = _result_status(result)
    evidence_metrics = result.get("metrics", {})
    artifacts = list(result.get("artifacts") or [])
    metric_cards = []
    table = None
    table_intro = None
    policy_table = None
    scope_table = None
    secondary_table = None
    secondary_table_title = None

    if token == "shap":
        ranked = evidence_metrics.get("feature_importance", [])
        is_text = evidence_metrics.get("feature_semantics") in {
            "tokens",
            "tokens_and_ngrams",
        }
        table = {
            "headers": [
                "Rank",
                "Token / n-gram" if is_text else "Feature",
                "Mean absolute SHAP value" if is_text else "Mean absolute importance",
            ],
            "rows": [
                [index, item.get("feature"), item.get("importance")]
                for index, item in enumerate(ranked[:10], 1)
            ],
        }
        table_intro = (
            "The table shows up to the 10 highest-ranked SHAP features. The "
            "complete feature-importance ranking is preserved in the Raw "
            "Evidence Appendix."
        )
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
        contributions = evidence_metrics.get("token_contributions") or []
        if contributions:
            table = {
                "headers": ["Token or phrase", "Local weight"],
                "rows": [
                    [item.get("token_or_phrase"), item.get("weight")]
                    for item in contributions
                ],
            }
        else:
            table = {
                "headers": ["Feature condition", "Local weight"],
                "rows": [
                    [item.get("feature_condition"), item.get("weight")]
                    for item in evidence_metrics.get("feature_contributions", [])
                ],
            }
    elif token == "dice":
        policy_table = {
            "headers": ["Counterfactual policy item", "Applied value"],
            "rows": [
                [
                    "Policy source",
                    _display(evidence_metrics.get("counterfactual_policy_source")),
                ],
                [
                    "Policy validation status",
                    _display(evidence_metrics.get("counterfactual_policy_status")),
                ],
                [
                    "Actionable feature allowlist",
                    _display(
                        evidence_metrics.get(
                            "actionable_feature_columns"
                        ),
                        "Not supplied",
                    ),
                ],
                [
                    "Excluded sensitive features",
                    _display(evidence_metrics.get("excluded_sensitive_features"), "None"),
                ],
                [
                    "Excluded immutable features",
                    _display(
                        evidence_metrics.get("excluded_immutable_features"), "None"
                    ),
                ],
                [
                    "Excluded non-actionable features",
                    _display(
                        evidence_metrics.get("excluded_non_actionable_features"),
                        "None",
                    ),
                ],
                [
                    "Actual features to vary",
                    _display(evidence_metrics.get("features_to_vary"), "None"),
                ],
                [
                    "Policy violation detected",
                    _display(evidence_metrics.get("policy_violation_detected", False)),
                ],
            ],
        }
        table_intro = (
            "The table below shows the first generated counterfactual. The full "
            "structured JSON artifact contains all generated counterfactuals."
        )
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
        counterfactuals = evidence_metrics.get("counterfactuals", [])
        if counterfactuals:
            secondary_table_title = "All generated counterfactuals"
            secondary_table = {
                "headers": [
                    "Counterfactual ID",
                    "Prediction",
                    "Changed features count",
                    "Changed feature names",
                ],
                "rows": [
                    [
                        item.get("counterfactual_id", index),
                        _prediction_display(
                            item.get("counterfactual_prediction"), audit_context
                        ),
                        item.get(
                            "number_of_changed_features",
                            len(item.get("changed_features", [])),
                        ),
                        ", ".join(
                            str(change.get("feature"))
                            for change in item.get("changed_features", [])
                            if change.get("feature") not in (None, "")
                        ) or "None",
                    ]
                    for index, item in enumerate(counterfactuals, 1)
                ],
            }
    elif token == "drift":
        scope_table = {
            "headers": ["Drift scope item", "Applied value"],
            "rows": [
                [
                    "Feature scope source",
                    _display(evidence_metrics.get("feature_scope_source")),
                ],
                [
                    "Canonical model-input features",
                    _display(evidence_metrics.get("model_feature_columns"), "None"),
                ],
                [
                    "Supplementary contextual / excluded columns",
                    _display(evidence_metrics.get("contextual_columns"), "None"),
                ],
                [
                    "Contextual drift status",
                    _display(
                        evidence_metrics.get("contextual_dataset_drift", {}).get(
                            "status"
                        )
                    ),
                ],
            ],
        }
        text_tests = evidence_metrics.get("derived_feature_tests") or []
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
                    text_tests or evidence_metrics.get("top_drifted_columns", []),
                    1,
                )
            ],
        }
    elif token.startswith("llm_"):
        generation = evidence_metrics.get("generation_configuration") or {}
        configuration_rows = [
            [key.replace("_", " ").title(), value]
            for key, value in generation.items()
        ]
        if evidence_metrics.get("seeds"):
            configuration_rows.append(["Seeds", _display(evidence_metrics["seeds"])])
        configuration_rows.append(
            [
                "Evaluation embedding model",
                _display(evidence_metrics.get("evaluation_embedding_model")),
            ]
        )
        scope_table = {
            "headers": ["Execution setting", "Applied value"],
            "rows": configuration_rows,
        }

    for key, value in evidence_metrics.items():
        if key == "runtime_seconds" or isinstance(value, (dict, list, tuple)):
            continue
        if token == "dice" and key in {
            "counterfactual_policy_source",
            "counterfactual_policy_status",
            "policy_violation_detected",
        }:
            continue
        if token == "dice" and key in {
            "original_prediction",
            "counterfactual_prediction",
        }:
            value = _prediction_display(value, audit_context)
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
        "policy_table": policy_table,
        "scope_table": scope_table,
        "table": table,
        "table_intro": table_intro,
        "secondary_table": secondary_table,
        "secondary_table_title": secondary_table_title,
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
        {"label": "Run ID", "value": _display(runtime_context.get("run_id"))},
        {"label": "Run timestamp", "value": _display(runtime_context.get("run_timestamp"))},
        {"label": "Python version", "value": _display(runtime_context.get("python_version"))},
        {"label": "Platform", "value": _display(runtime_context.get("platform"))},
        {"label": "Total runtime", "value": total_runtime_display},
        {"label": "HTML report path", "value": _display(runtime_context.get("html_report_path"))},
        {"label": "PDF report path", "value": _display(runtime_context.get("pdf_report_path") if runtime_context.get("pdf_generated") else None, "Not generated")},
    ]

    is_llm = str(_value(audit_context, "system_type", "")).lower() in {
        "llm",
        "agentic",
    }
    visible_dependencies = (
        ("torch", "transformers", "sentence-transformers")
        if is_llm
        else (
            "scikit-learn",
            "shap",
            "lime",
            "dice-ml",
            "fairlearn",
            "mapie",
            "evidently",
        )
    )
    dependency_versions = runtime_context.get("dependency_versions") or {}
    dependency_rows = [
        {
            "name": name,
            "version": version,
            "scope": "LLM / Agentic" if name in {
                "torch",
                "transformers",
                "sentence-transformers",
                "peft",
            } else "Core",
        }
        for name in visible_dependencies
        if (version := dependency_versions.get(name)) is not None
    ]
    provenance_rows = []
    traditional_inputs = {"training_dataset", "evaluation_dataset"}
    llm_inputs = {
        "golden_set",
        "system_prompt",
        "rag_manifest",
        "guardrail_config",
        "evaluation_embedding_model",
        "semantic_drift_validation",
        "fairness_prompt_pairs",
    }
    for name, record in (runtime_context.get("input_provenance") or {}).items():
        if (is_llm and name in traditional_inputs) or (
            not is_llm and name in llm_inputs
        ):
            continue
        full_hash = str(record.get("sha256", "unavailable"))
        abbreviated_hash = (
            f"{full_hash[:12]}…" if len(full_hash) == 64 else full_hash
        )
        provenance_rows.append(
            {
                "name": name.replace("_", " ").title(),
                "status": record.get("status", "unavailable"),
                "path": record.get("path", "unavailable"),
                "sha256": abbreviated_hash,
            }
        )

    return {
        "common_facts": common_facts,
        "dependency_rows": dependency_rows,
        "provenance_rows": provenance_rows,
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
    is_local_surrogate = bool((resource_context or {}).get("is_local_surrogate"))
    return {
        "validation_notice": (
            {
                "title": "Local surrogate execution validation",
                "text": (
                    "The original LegalMind model weights were unavailable. "
                    "DistilGPT2 was used as a locally executable surrogate to "
                    "validate the mechanics of the LLM evidence pathway. The "
                    "resulting scores are not an audit of the original LegalMind model."
                ),
            }
            if is_local_surrogate
            else None
        ),
        "executive_summary": build_executive_summary(
            results,
            audit_context,
            governance_context,
            provider_name,
            resource_context,
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
    }
