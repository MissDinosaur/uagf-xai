"""
Constraint-Based Evidence Planner (CBEP).
Selects the minimal evidence methods required for auditing.
"""


def select_methods(risk_level):

    if risk_level == "minimal":
        return ["shap"]

    if risk_level == "limited":
        return ["shap", "fairness"]

    if risk_level == "high":
        return ["shap", "fairness", "uncertainty", "drift"]

    return []
