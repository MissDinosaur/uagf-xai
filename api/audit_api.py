from __future__ import annotations

import time

from planner.cbep import plan_evidence
from pipeline.executor import execute
from report.report_generator import generate_report
from resources.loader import ResourceLoader
from pipeline.evidence_normalizer import normalize_evidence_results


def _is_llm_contract(audit_context) -> bool:
    system_type = str(getattr(audit_context, "system_type", "") or "").strip().lower()
    task_type = str(getattr(audit_context, "task_type", "") or "").strip().lower()
    return task_type == "llm_generation" or system_type in {"llm", "agentic"}


def _split_evaluation_frame(evaluation_dataset, target_column: str):
    if hasattr(evaluation_dataset, "columns") and target_column in evaluation_dataset.columns:
        X = evaluation_dataset.drop(columns=[target_column])
        y = evaluation_dataset[target_column]
        return X, y
    return evaluation_dataset, None


def _load_s5_resources(audit_context):
    resources = ResourceLoader.load_bundle(audit_context)

    if _is_llm_contract(audit_context):
        golden_set = resources.golden_dataset
        if golden_set is None:
            raise ValueError(
                "LLM / Agentic contract requires audit_context.golden_set_uri "
                "or an equivalent golden-set resource."
            )
        sensitive_features = list(audit_context.sensitive_feature_columns or [])
        return resources, golden_set, None, sensitive_features

    X, y = _split_evaluation_frame(
        resources.evaluation_dataset,
        getattr(audit_context, "target_column", "target"),
    )
    sensitive_features = list(audit_context.sensitive_feature_columns or [])
    return resources, X, y, sensitive_features


def _build_resource_context(audit_context, resource_bundle):
    model = getattr(resource_bundle, "model", None)
    model_metadata = getattr(resource_bundle, "model_metadata", {}) or {}

    model_class = type(model).__name__ if model is not None else None
    if getattr(model, "status", None) == "metadata_only":
        artifact_kind = "Metadata-only LLM model folder"
    elif getattr(audit_context, "model_format", None) == "model_directory":
        artifact_kind = "Model directory"
    else:
        artifact_kind = "Single model file"
    model_container = (
        "Sklearn model bundle"
        if model_class == "EncodedSklearnModel"
        else model_class or "Not available"
    )

    return {
        "golden_set_uri": getattr(audit_context, "golden_set_uri", None),
        "golden_set": getattr(resource_bundle, "golden_dataset", None),
        "system_prompt": getattr(resource_bundle, "system_prompt", None),
        "rag_manifest": getattr(resource_bundle, "rag_manifest", None),
        "guardrail_config": getattr(resource_bundle, "guardrail_config", None),
        "model_metadata": model_metadata,
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
    generate_pdf=True,
    run_started_at=None,
):
    """
    Run the evidence pipeline against explicitly supplied model/data inputs.
    """
    pipeline_started_at = run_started_at or time.perf_counter()
    methods, cbep_trace = plan_evidence(audit_context, governance_context)
    print("Selected modules:", methods)
    provider_name = getattr(audit_context, "provider_name", None)
    output_namespace = getattr(audit_context, "output_namespace", None)
    resource_context = None
    if resource_bundle is not None:
        resource_context = _build_resource_context(
            audit_context,
            resource_bundle,
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
        output_namespace=output_namespace,
        resource_context=resource_context,
        task_type=audit_context.task_type,
        drift_reference_data=drift_reference_data,
        target_column=getattr(audit_context, "target_column", None),
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
        output_namespace=output_namespace,
        audit_context=audit_context,
        governance_context=governance_context,
        resource_context=resource_context,
        generate_pdf=generate_pdf,
        runtime_context={
            "total_runtime_seconds": round(total_runtime_seconds, 6),
            "measurement_scope": (
                "Resource loading, CBEP planning, evidence execution, and evidence "
                "normalization; report rendering and PDF export are excluded."
            ),
        },
    )

    return results


def audit(audit_context, governance_context=None, generate_pdf=True):
    """
    Run the S5-driven workflow.

    This path requires real S5-provided resources. If the model artifact or
    datasets are unavailable, the failure is surfaced directly to the caller.
    """
    run_started_at = time.perf_counter()
    resource_bundle, X, y, sensitive_features = _load_s5_resources(audit_context)
    print("Mode: real S5 resources")

    return audit_with_detailed_data(
        model=resource_bundle.model,
        X=X,
        y=y,
        audit_context=audit_context,
        governance_context=governance_context,
        sensitive_features=sensitive_features,
        resource_bundle=resource_bundle,
        generate_pdf=generate_pdf,
        run_started_at=run_started_at,
    )
