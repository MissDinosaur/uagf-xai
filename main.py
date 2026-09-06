"""
Main execution pipeline for the UAGF-XAI audit toolkit.

Two explicit execution modes are supported: formal S4/S5 audit resources and
an S6-owned local LLM execution-validation case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapters.s4_governance_adapter import GovernanceAdapter
from adapters.s5_audit_adapter import AuditAdapter
from api.audit_api import audit
from resources.local_llm_validation_config import LocalLLMValidationConfig


# Default parameters for the S6 local DistilGPT2 execution-validation case
DEFAULT_LOCAL_CONFIG = (
    "data/04b_local_distilgpt2_llm/s6_local_llm_validation_config.json"
)
DEFAULT_LOCAL_S4 = (
    "data/04b_local_distilgpt2_llm/s4_local_llm_validation_context.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run UAGF-XAI audit workflow")
    parser.add_argument(
        "--mode",
        choices=["s5", "local"],
        default="s5",
        help="Use the formal S5 contract or the S6 local LLM validation config.",
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
        "--local-config",
        default=DEFAULT_LOCAL_CONFIG,
        metavar="FILE",
        help="Path to the S6-owned local LLM validation configuration.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Generate the HTML report only and skip automatic PDF export.",
    )
    return parser.parse_args()


def _load_s4_governance_context(path: str | None):
    if not path:
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded S4 governance JSON : {path}")
    context = GovernanceAdapter.from_cgsa_report(data)
    context.source_json_path = Path(path).as_posix()
    return context


def _load_s5_audit_context(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded S5 audit JSON      : {path}")
    context = AuditAdapter.from_audit_report(data)
    context.source_json_path = Path(path).as_posix()
    return context


def main():
    args = parse_args()
    if args.mode == "local":
        local_case = LocalLLMValidationConfig.load(args.local_config)
        s4_path = args.s4_json or DEFAULT_LOCAL_S4
        governance_context = _load_s4_governance_context(s4_path)
        audit_context = local_case.audit_context
        print(f"Loaded S6 local config    : {local_case.source_path}")
        results = audit(
            audit_context,
            governance_context,
            generate_pdf=not args.no_pdf,
            runtime_mode="local",
            llm_evidence_config=local_case.llm_evidence_config,
            local_validation_config_path=local_case.source_path,
        )
    else:
        if not args.s5_json:
            raise ValueError("--s5-json is required in --mode s5")

        governance_context = _load_s4_governance_context(args.s4_json)
        audit_context = _load_s5_audit_context(args.s5_json)
        print()
        print("Mode: S5-driven resources")
        print(f"Risk tier    : {audit_context.risk_tier}")
        print(f"Application domain: {audit_context.application_domain}")
        print(f"Task type    : {audit_context.task_type}")
        print(f"Target column: {audit_context.target_column}")
        print(f"Model artifact URI: {audit_context.model_artifact_uri}")
        print(f"Model artifact kind: {audit_context.model_artifact_kind}")

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

# S6 local DistilGPT2 execution-validation case
python main.py --mode local

# S5-driven mode with governance and audit JSON files
python main.py --mode s5 \\
    --s4-json data/03_harbourlogistik_gmbh/s4_harbourlogistik-harboursense-001.json \\
    --s5-json data/03_harbourlogistik_gmbh/s5_harbourlogistik_gmbh_audit_state.json
"""
