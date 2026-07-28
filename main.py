"""
Main execution pipeline for the UAGF-XAI audit toolkit.

Two execution paths are supported:

1. Local validation mode
   - enabled with --system-type
   - uses locally runnable models and datasets

2. json-files-driven mode
   - omit --system-type
   - loads audit metadata from the GovernanceContext JSON and AuditContext JSON 
     and resolves resources from them
"""

from __future__ import annotations

import argparse
import json

from adapters.s4_governance_adapter import GovernanceAdapter
from adapters.s5_audit_adapter import AuditAdapter, AuditContext
from api.audit_api import audit, audit_with_detailed_data
from models.german_credit import train_model
from models.llm_validation import load_llm_validation_domain


def parse_args():
    parser = argparse.ArgumentParser(description="Run UAGF-XAI audit workflow")
    parser.add_argument(
        "--system-type",
        choices=["traditional", "llm", "agentic"],
        default=None,
        help=(
            "Run local validation mode instead of loading model/data from S5. "
            "If omitted, the workflow uses S5-driven resources."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default="gpt2",
        help="HuggingFace model ID for local LLM/Agentic validation mode",
    )
    parser.add_argument(
        "--s4-json",
        default=None,
        metavar="FILE",
        help="Path to S4 governance JSON file",
    )
    parser.add_argument(
        "--s5-json",
        default=None,
        metavar="FILE",
        help="Path to S5 audit JSON file",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Generate the HTML report only and skip automatic PDF export.",
    )
    return parser.parse_args()


def _load_governance_context(path: str | None):
    if not path:
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded S4 governance JSON : {path}")
    return GovernanceAdapter.from_cgsa_report(data)


def _load_s5_audit_context(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded S5 audit JSON      : {path}")
    return AuditAdapter.from_audit_report(data)


def _build_local_validation_audit_context(system_type: str) -> AuditContext:
    system_type = str(system_type).strip().lower()

    if system_type in {"llm", "agentic"}:
        audit_context = AuditContext(
            system_type=system_type,
            modality="text",
            application_domain="local_validation",
            risk_tier="high",
            applicable_articles=[],
            sensitive_feature_columns=[],
            target_column="prompt",
            positive_label=None,
            task_type="llm_generation",
            provider_name="local_validation",
        )
        return audit_context

    audit_context = AuditContext(
        system_type=system_type,
        modality="tabular",
        application_domain="local_validation",
        risk_tier="high",
        applicable_articles=[],
        sensitive_feature_columns=[],
        target_column="target",
        positive_label=1,
        task_type="binary_classification",
        provider_name="local_validation",
    )
    return audit_context


def _load_local_validation_resources(system_type: str, llm_model: str):
    system_type = str(system_type).strip().lower()
    if system_type in {"llm", "agentic"}:
        model, llm_payload = load_llm_validation_domain(model_name=llm_model)
        return model, llm_payload, None, []

    return train_model()


def main():
    args = parse_args()
    governance_context = _load_governance_context(args.s4_json)

    if args.system_type is not None:
        audit_context = _build_local_validation_audit_context(args.system_type)
        print("Mode: local validation resources")
        print(f"System type  : {audit_context.system_type}")
        print(f"Task type    : {audit_context.task_type}")

        model, X_test, y_test, sensitive_features = _load_local_validation_resources(
            args.system_type,
            args.llm_model,
        )

        results = audit_with_detailed_data(
            model=model,
            X=X_test,
            y=y_test,
            audit_context=audit_context,
            governance_context=governance_context,
            sensitive_features=sensitive_features,
            generate_pdf=not args.no_pdf,
        )
    else:
        if not args.s5_json:
            raise ValueError("S5 JSON is required when --system-type is not provided")

        audit_context = _load_s5_audit_context(args.s5_json)
        print("\\n")
        print("Mode: S5-driven resources")
        print(f"Risk tier    : {audit_context.risk_tier}")
        print(f"Application domain: {audit_context.application_domain}")
        print(f"Task type    : {audit_context.task_type}")
        print(f"Target column: {audit_context.target_column}")
        print(f"Model artifact URI: {audit_context.model_artifact_uri}")

        results = audit(
            audit_context,
            governance_context,
            generate_pdf=not args.no_pdf,
        )

    print("\nFinal Results:")
    print(results)


if __name__ == "__main__":
    main()


"""
Example usage:

# Local validation mode with mocked German Credit dataset
python main.py --system-type traditional

# Local validation mode with mocked LLM/Agentic system
python main.py --system-type llm --llm-model gpt2

# S5-driven mode with governance and audit JSON files
python main.py \\
    --s4-json data/03_harbourlogistik_gmbh/s4_harbourlogistik-harboursense-001.json \\
    --s5-json data/03_harbourlogistik_gmbh/s5_harbourlogistik_gmbh_audit_state.json
"""
