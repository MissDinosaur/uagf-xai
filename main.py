"""
Main execution pipeline for the UAGF-XAI audit toolkit.
This script coordinates model training, planning, and evidence generation.
"""

import argparse

from models.german_credit import train_model
from planner.cbep import select_methods
from api.audit_api import audit


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UAGF-XAI audit workflow"
    )
    parser.add_argument(
        "--system-type",
        choices=["traditional", "llm", "agentic"],
        default="traditional",
        help="Type of target system to audit",
    )
    parser.add_argument(
        "--risk-level",
        choices=["minimal", "limited", "high"],
        default="high",
        help="Risk level used by CBEP planner",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt2",
        help="HuggingFace model ID for LLM/Agentic mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Choose system type: "traditional", "llm", or "agentic"
    system_type = args.system_type
    risk_level = args.risk_level

    if system_type in {"llm", "agentic"}:
        from models.llm_validation import load_llm_validation_domain

        # Validation Domain 5: local open-source LLM via HuggingFace
        model, X_test = load_llm_validation_domain(model_name=args.llm_model)
        y_test = None
    else:
        # Traditional ML systems
        model, X_test, y_test = train_model()

    # Planner selects which modules should run
    selected_methods = select_methods(risk_level, system_type=system_type)

    print("Selected modules:", selected_methods)

    results = audit(
        model=model,
        X=X_test,
        y=y_test,
        risk_level=risk_level,
        system_type=system_type,
    )

    print("\nFinal Results:")
    print(results)


if __name__ == "__main__":
    main()


"""
python main.py --system-type traditional --risk-level high
python main.py --system-type llm --risk-level high --llm-model gpt2
"""