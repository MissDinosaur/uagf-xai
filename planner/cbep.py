"""
Constraint-Based Evidence Planner (CBEP).
Selects the minimal evidence methods required for auditing.
"""


def select_methods(risk_level, system_type="traditional"):

    system_type_normalized = str(system_type).strip().lower()
    is_llm_system = system_type_normalized in {"llm", "agentic"}

    if is_llm_system:
        if risk_level == "minimal":
            return ["llm_explainability"]

        if risk_level == "limited":
            return ["llm_explainability", "llm_fairness"]

        if risk_level == "high":
            return [
                "llm_explainability",
                "llm_fairness",
                "llm_uncertainty",
                "llm_drift",
            ]

        return []

    if risk_level == "minimal":
        return ["shap"]

    if risk_level == "limited":
        return ["shap", "fairness"]

    if risk_level == "high":
        return ["shap", "fairness", "uncertainty", "drift", "dice"]

    return []
