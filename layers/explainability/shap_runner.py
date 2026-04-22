"""
Explainability module using SHAP.
"""

import shap
import matplotlib.pyplot as plt
import os

def run_shap(model, X):

    os.makedirs("outputs/shap", exist_ok=True)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("outputs/shap/shap_summary.png")
    plt.close()

    # Feature importance（mean |SHAP|）
    importance = abs(shap_values.values).mean(axis=0)

    result = {
        "type": "explainability",
        "method": "SHAP",
        "feature_importance": importance.tolist(),
        "plot": "outputs/shap/shap_summary.png"
    }

    return result
