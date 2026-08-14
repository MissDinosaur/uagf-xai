from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from planner.cbep import plan_evidence
from pipeline.executor import execute
from report.report_generator import generate_report
from resources.loader import ResourceLoader
from pipeline.evidence_normalizer import normalize_evidence_results


def _is_llm_contract(audit_context) -> bool:
    system_type = str(getattr(audit_context, "system_type", "") or "").strip().lower()
    return system_type in {"llm", "agentic"}


@dataclass
class EvaluationViews:
    evaluation_frame: pd.DataFrame
    X_model: Any
    y: pd.Series | None
    sensitive_data: pd.DataFrame | None
    feature_columns: list[str]
    sensitive_feature_columns: list[str]
    feature_scope_source: str


def _build_evaluation_views(
    evaluation_dataset,
    *,
    target_column,
    feature_columns,
    sensitive_feature_columns,
    modality,
    feature_scope_source=None,
) -> EvaluationViews:
    if not isinstance(evaluation_dataset, pd.DataFrame):
        raise TypeError("The S5 evaluation dataset must load as a pandas DataFrame.")

    evaluation_frame = evaluation_dataset.copy()
    if target_column and target_column not in evaluation_frame.columns:
        raise ValueError(
            f"Declared target column {target_column!r} is missing from the "
            "evaluation dataset."
        )
    y = evaluation_frame[target_column].copy() if target_column else None

    if feature_columns is None:
        model_columns = [
            column for column in evaluation_frame.columns if column != target_column
        ]
    else:
        missing = [
            column for column in feature_columns if column not in evaluation_frame.columns
        ]
        if missing:
            raise ValueError(
                "Declared model feature columns are missing from the evaluation "
                f"dataset: {missing}"
            )
        model_columns = list(feature_columns)

    normalized_modality = str(modality or "unknown").strip().lower()
    if normalized_modality == "text" and len(model_columns) != 1:
        raise ValueError(
            "Traditional text evaluation currently requires exactly one declared "
            f"model feature column; received {model_columns}."
        )
    X_model = evaluation_frame.loc[:, model_columns].copy()

    sensitive_columns = list(sensitive_feature_columns or [])
    missing_sensitive = [
        column for column in sensitive_columns if column not in evaluation_frame.columns
    ]
    if missing_sensitive:
        raise ValueError(
            "Declared sensitive feature columns are missing from the evaluation "
            f"dataset: {missing_sensitive}"
        )
    sensitive_data = (
        evaluation_frame.loc[:, sensitive_columns].copy()
        if sensitive_columns
        else None
    )
    return EvaluationViews(
        evaluation_frame=evaluation_frame,
        X_model=X_model,
        y=y,
        sensitive_data=sensitive_data,
        feature_columns=model_columns,
        sensitive_feature_columns=sensitive_columns,
        feature_scope_source=(
            feature_scope_source
            or (
                "audit_context_feature_columns"
                if feature_columns
                else "legacy_target_drop_fallback"
            )
        ),
    )


def _load_s5_resources(audit_context, llm_evidence_config=None):
    if llm_evidence_config is None:
        resources = ResourceLoader.load_bundle(audit_context)
    else:
        resources = ResourceLoader.load_bundle(audit_context, llm_evidence_config)

    if _is_llm_contract(audit_context):
        golden_set = resources.golden_dataset
        if golden_set is None:
            raise ValueError(
                "LLM / Agentic contract requires audit_context.golden_set_uri "
                "or an equivalent golden-set resource."
            )
        return resources, golden_set

    artifact_feature_columns = getattr(resources, "model_feature_columns", None)
    audit_feature_columns = getattr(audit_context, "feature_columns", None)
    if artifact_feature_columns:
        effective_feature_columns = list(artifact_feature_columns)
        feature_scope_source = "model_artifact_feature_cols"
    elif audit_feature_columns:
        effective_feature_columns = list(audit_feature_columns)
        feature_scope_source = "audit_context_feature_columns"
    else:
        effective_feature_columns = None
        feature_scope_source = "legacy_target_drop_fallback"

    views = _build_evaluation_views(
        resources.evaluation_dataset,
        target_column=getattr(audit_context, "target_column", None),
        feature_columns=effective_feature_columns,
        sensitive_feature_columns=audit_context.sensitive_feature_columns,
        modality=audit_context.modality,
        feature_scope_source=feature_scope_source,
    )
    return resources, views


def _build_resource_context(audit_context, resource_bundle, runtime_mode="s5"):
    model = getattr(resource_bundle, "model", None)
    model_metadata = getattr(resource_bundle, "model_metadata", {}) or {}
    evidence_config = getattr(resource_bundle, "llm_evidence_config", None)

    model_class = type(model).__name__ if model is not None else None
    if getattr(model, "status", None) == "metadata_only":
        artifact_kind = "Metadata-only LLM model folder"
    elif getattr(audit_context, "model_artifact_kind", None) == "directory":
        artifact_kind = "Model directory"
    else:
        artifact_kind = "Single model file"
    model_container = (
        "Sklearn model bundle"
        if model_class == "EncodedSklearnModel"
        else (
            "Fitted sklearn text pipeline"
            if model_class == "SklearnTextModelAdapter"
            else model_class or "Not available"
        )
    )

    return {
        "golden_set_uri": getattr(audit_context, "golden_set_uri", None),
        "golden_set": getattr(resource_bundle, "golden_dataset", None),
        "system_prompt": getattr(resource_bundle, "system_prompt", None),
        "rag_manifest": getattr(resource_bundle, "rag_manifest", None),
        "guardrail_config": getattr(resource_bundle, "guardrail_config", None),
        "evaluation_embedding_model": getattr(
            resource_bundle, "embedding_model", None
        ),
        "evaluation_embedding_metadata": getattr(
            resource_bundle, "evaluation_embedding_metadata", {}
        ),
        "semantic_drift_dataset": getattr(
            resource_bundle, "semantic_drift_dataset", None
        ),
        "fairness_prompt_pairs": getattr(
            resource_bundle, "fairness_prompt_pairs", None
        ),
        "runtime_mode": runtime_mode,
        "is_local_surrogate": runtime_mode == "local",
        "evaluation_embedding_model_name": getattr(
            evidence_config, "evaluation_embedding_model", None
        ),
        "model_metadata": model_metadata,
        "feature_columns": getattr(audit_context, "feature_columns", None),
        "model_feature_columns": getattr(resource_bundle, "model_feature_columns", None),
        "model_feature_scope_source": getattr(
            resource_bundle, "model_feature_scope_source", None
        ),
        "sensitive_feature_columns": getattr(
            audit_context,
            "sensitive_feature_columns",
            [],
        ),
        "model_artifact_kind": getattr(audit_context, "model_artifact_kind", None),
        "model_status": getattr(model, "status", model_metadata.get("status")),
        "model_is_loadable": getattr(
            model,
            "is_loadable",
            model_metadata.get("is_loadable", True),
        ),
        "model_reason": getattr(model, "reason", model_metadata.get("reason")),
        "resolved_path": getattr(
            model,
            "resolved_path",
            model_metadata.get("resolved_artifact_path"),
        ),
        "model_type": getattr(model, "model_type", model_metadata.get("model_type")),
        "model_framework": getattr(
            model,
            "model_framework",
            model_metadata.get("model_framework"),
        ),
        "model_entrypoint": getattr(
            model,
            "model_entrypoint",
            model_metadata.get("model_entrypoint"),
        ),
        "loaded_metadata_files": getattr(
            model,
            "loaded_metadata_files",
            model_metadata.get("loaded_metadata_files", []),
        ),
        "artifact_kind": artifact_kind,
        "model_container": model_container,
        "model_class": model_class,
        "load_warnings": getattr(model, "_uagf_load_warnings", []),
        "input_adapter": model_metadata.get("input_adapter"),
        "text_feature_column": model_metadata.get("text_feature_column"),
        "vectorizer_class": model_metadata.get("vectorizer_class"),
        "estimator_class": model_metadata.get("estimator_class"),
        "vocabulary_size": model_metadata.get("vocabulary_size"),
        "training_dataset_loaded": (
            getattr(resource_bundle, "training_dataset", None) is not None
        ),
        "training_dataset_rows": _safe_record_count(
            getattr(resource_bundle, "training_dataset", None)
        ),
        "evaluation_dataset_loaded": (
            getattr(resource_bundle, "evaluation_dataset", None) is not None
        ),
        "evaluation_dataset_rows": _safe_record_count(
            getattr(resource_bundle, "evaluation_dataset", None)
        ),
        "golden_set_loaded": (
            getattr(resource_bundle, "golden_dataset", None) is not None
        ),
        "golden_set_records": _safe_record_count(
            getattr(resource_bundle, "golden_dataset", None)
        ),
    }


def _safe_record_count(value):
    """Return a resource row count when it can be determined safely."""
    if value is None or isinstance(value, (str, bytes)):
        return None
    if hasattr(value, "shape"):
        try:
            return int(value.shape[0])
        except (IndexError, TypeError, ValueError):
            pass
    if isinstance(value, dict):
        for key in ("golden_set", "records", "items", "data", "examples"):
            candidate = value.get(key)
            if isinstance(candidate, (list, tuple)):
                return len(candidate)
        candidate_lengths = [
            len(candidate)
            for candidate in value.values()
            if isinstance(candidate, (list, tuple))
        ]
        return max(candidate_lengths) if candidate_lengths else None
    try:
        return len(value)
    except (TypeError, ValueError):
        return None


def audit_with_detailed_data(
    model,
    X,
    y,
    audit_context,
    governance_context=None,
    sensitive_features=None,
    resource_bundle=None,
    evaluation_frame=None,
    sensitive_data=None,
    feature_columns=None,
    feature_scope_source=None,
    generate_pdf=True,
    run_started_at=None,
    runtime_mode="s5",
    local_validation_config_path=None,
):
    """
    Run the evidence pipeline against explicitly supplied model/data inputs.
    """
    pipeline_started_at = run_started_at or time.perf_counter()
    methods, cbep_trace = plan_evidence(audit_context, governance_context)
    print("Selected modules:", methods)
    provider_name = getattr(audit_context, "provider_name", None)
    resource_context = None
    if resource_bundle is not None:
        resource_context = _build_resource_context(
            audit_context,
            resource_bundle,
            runtime_mode,
        )
    drift_reference_data = (
        getattr(resource_bundle, "training_dataset", None)
        if resource_bundle is not None
        else None
    )

    results = execute(
        model,
        X,
        y,
        methods,
        system_type=audit_context.system_type,
        sensitive_features=sensitive_features,
        provider_name=provider_name,
        resource_context=resource_context,
        task_type=audit_context.task_type,
        drift_reference_data=drift_reference_data,
        evaluation_frame=evaluation_frame,
        sensitive_data=sensitive_data,
        feature_columns=feature_columns,
        feature_scope_source=(
            feature_scope_source
            or (
                "audit_context_feature_columns"
                if getattr(audit_context, "feature_columns", None)
                else "legacy_target_drop_fallback"
            )
        ),
        modality=audit_context.modality,
        positive_label=getattr(audit_context, "positive_label", 1),
        target_column=getattr(audit_context, "target_column", None),
        actionable_feature_columns=getattr(
            audit_context, "actionable_feature_columns", None
        ),
        immutable_feature_columns=getattr(audit_context, "immutable_feature_columns", []),
    )

    results["_cbep_trace"] = {
        "type": "cbep_planning_trace",
        "method": "CBEP (Constraint-Based Evidence Planner)",
        **cbep_trace,
    }
    results = normalize_evidence_results(
        results,
        cbep_trace=cbep_trace,
        task_type=audit_context.task_type,
    )

    total_runtime_seconds = time.perf_counter() - pipeline_started_at

    generate_report(
        results,
        audit_context.risk_tier,
        provider_name=provider_name,
        audit_context=audit_context,
        governance_context=governance_context,
        resource_context=resource_context,
        generate_pdf=generate_pdf,
        runtime_context={
            "total_runtime_seconds": round(total_runtime_seconds, 6),
            "runtime_mode": runtime_mode,
            "local_validation_config_path": local_validation_config_path,
            "llm_evidence_config": getattr(
                resource_bundle, "llm_evidence_config", None
            ),
            "measurement_scope": (
                "Resource loading, CBEP planning, evidence execution, and evidence "
                "normalization; report rendering and PDF export are excluded."
            ),
        },
    )

    return results


def audit(
    audit_context,
    governance_context=None,
    generate_pdf=True,
    *,
    runtime_mode="s5",
    llm_evidence_config=None,
    local_validation_config_path=None,
):
    """
    Run an S5 contract or an explicitly configured S6 local validation case.

    Resource failures are surfaced directly; no synthetic inference fallback is used.
    """
    run_started_at = time.perf_counter()
    resource_bundle, payload = _load_s5_resources(
        audit_context, llm_evidence_config
    )
    if runtime_mode == "local":
        print("Mode: S6 local execution fallback")
    else:
        print("Mode: real S5 resources")

    if _is_llm_contract(audit_context):
        X = payload
        y = None
        sensitive_features = list(audit_context.sensitive_feature_columns or [])
        views = None
    else:
        views = payload
        X = views.X_model
        y = views.y
        sensitive_features = views.sensitive_feature_columns

    return audit_with_detailed_data(
        model=resource_bundle.model,
        X=X,
        y=y,
        audit_context=audit_context,
        governance_context=governance_context,
        sensitive_features=sensitive_features,
        resource_bundle=resource_bundle,
        evaluation_frame=views.evaluation_frame if views else None,
        sensitive_data=views.sensitive_data if views else None,
        feature_columns=views.feature_columns if views else None,
        feature_scope_source=views.feature_scope_source if views else None,
        generate_pdf=generate_pdf,
        run_started_at=run_started_at,
        runtime_mode=runtime_mode,
        local_validation_config_path=local_validation_config_path,
    )
