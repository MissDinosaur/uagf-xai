from __future__ import annotations

from planner.cbep import plan_evidence
from pipeline.executor import execute
from report.report_generator import generate_report
from resources.loader import ResourceLoader


def _split_evaluation_frame(evaluation_dataset, target_column: str):
    if hasattr(evaluation_dataset, "columns") and target_column in evaluation_dataset.columns:
        X = evaluation_dataset.drop(columns=[target_column])
        y = evaluation_dataset[target_column]
        return X, y
    return evaluation_dataset, None


def _load_real_s5_resources(audit_context):
    resources = ResourceLoader.load_bundle(audit_context)
    X, y = _split_evaluation_frame(
        resources.evaluation_dataset,
        getattr(audit_context, "target_column", "target"),
    )
    sensitive_features = list(audit_context.sensitive_feature_columns or [])
    return resources.model, X, y, sensitive_features


def audit_with_detailed_data(
    model,
    X,
    y,
    audit_context,
    governance_context=None,
    sensitive_features=None,
):
    """
    Run the evidence pipeline against explicitly supplied model/data inputs.
    """
    methods, cbep_trace = plan_evidence(audit_context, governance_context)
    print("Selected modules:", methods)
    provider_name = getattr(audit_context, "provider_name", None)
    output_namespace = getattr(audit_context, "output_namespace", None)

    results = execute(
        model,
        X,
        y,
        methods,
        system_type=audit_context.system_type,
        sensitive_features=sensitive_features,
        provider_name=provider_name,
        output_namespace=output_namespace,
    )

    results["_cbep_trace"] = {
        "type": "cbep_planning_trace",
        "method": "CBEP (Constraint-Based Evidence Planner)",
        **cbep_trace,
    }

    report_path = generate_report(
        results,
        audit_context.risk_tier,
        provider_name=provider_name,
        output_namespace=output_namespace,
    )
    print("Report generated at:", report_path)

    return results


def audit(audit_context, governance_context=None):
    """
    Run the S5-driven workflow.

    This path requires real S5-provided resources. If the model artifact or
    datasets are unavailable, the failure is surfaced directly to the caller.
    """
    model, X, y, sensitive_features = _load_real_s5_resources(audit_context)
    print("Mode: real S5 resources")

    return audit_with_detailed_data(
        model=model,
        X=X,
        y=y,
        audit_context=audit_context,
        governance_context=governance_context,
        sensitive_features=sensitive_features,
    )
