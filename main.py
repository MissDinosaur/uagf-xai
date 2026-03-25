"""
Main execution pipeline for the UAGF-XAI audit toolkit.
This script coordinates model training, planning, and evidence generation.
"""

from models.model_pipeline import train_model
from planner.cbep import select_methods

from layers.explainability.shap_runner import run_shap
from layers.fairness.fairlearn_runner import run_fairness
from layers.uncertainty.mapie_runner import run_uncertainty
from layers.drift.evidently_runner import run_drift


def main():

    # Train or load the ML model
    model, X_test, y_test = train_model()

    # Define risk level of the AI system
    risk_level = "high"

    # Planner selects which modules should run
    selected_methods = select_methods(risk_level)

    print("Selected modules:", selected_methods)

    if "shap" in selected_methods:
        run_shap(model, X_test)

    if "fairness" in selected_methods:
        run_fairness(model, X_test, y_test)

    if "uncertainty" in selected_methods:
        run_uncertainty(model, X_test, y_test)

    if "drift" in selected_methods:
        run_drift(X_test)


if __name__ == "__main__":
    main()
