"""
Explainability module using SHAP.
"""

import shap


def run_shap(model, X):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X)

    print("SHAP explanation generated.")
