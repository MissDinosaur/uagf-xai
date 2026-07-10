from layers.explainability.shap_runner import run_shap
from layers.fairness.fairlearn_runner import run_fairness
from layers.uncertainty.mapie_runner import run_uncertainty
from layers.drift.evidently_runner import run_drift
from layers.explainability.dice_runner import run_dice
from layers.llm.llm_explainability_runner import run_llm_explainability
from layers.llm.llm_fairness_runner import run_llm_fairness
from layers.llm.llm_uncertainty_runner import run_llm_uncertainty
from layers.llm.llm_drift_runner import run_llm_drift


def _resolve_model_views(model, X):
    """
    Keep raw and model-ready views separate.

    X_raw is preserved for reporting, fairness and drift logic.
    X_model is derived only when the model wrapper exposes prepare_input().
    """
    if hasattr(model, "prepare_input"):
        return X, model.prepare_input(X), getattr(model, "estimator", model)
    return X, X, model


def execute(
    model,
    X,
    y,
    methods,
    system_type="traditional",
    sensitive_features=None,
    provider_name=None,
    output_namespace="audit",
):

    results = {}
    system_type_normalized = str(system_type).strip().lower()
    is_llm_system = system_type_normalized in {"llm", "agentic"}
    X_raw, X_model, estimator = _resolve_model_views(model, X)

    if is_llm_system:
        if "llm_explainability" in methods:
            results["llm_explainability"] = run_llm_explainability(model, X_raw)

        if "llm_fairness" in methods:
            results["llm_fairness"] = run_llm_fairness(model, X_raw)

        if "llm_uncertainty" in methods:
            results["llm_uncertainty"] = run_llm_uncertainty(model, X_raw)

        if "llm_drift" in methods:
            results["llm_drift"] = run_llm_drift(X_raw)

        return results

    if "shap" in methods:
        results["explainability"] = run_shap(
            model,
            X_raw,
            X_model=X_model,
            estimator=estimator,
            provider_name=provider_name,
            output_namespace=output_namespace,
        )

    if "fairness" in methods:
        results["fairness"] = run_fairness(
            model,
            X_raw,
            y,
            sensitive_features=sensitive_features,
        )

    if "uncertainty" in methods:
        results["uncertainty"] = run_uncertainty(model, X_raw, y)

    if "drift" in methods:
        results["drift"] = run_drift(X_raw)

    if "dice" in methods:
        results["counterfactual"] = run_dice(
            model,
            X_raw,
            y,
            provider_name=provider_name,
            output_namespace=output_namespace,
        )

    return results
