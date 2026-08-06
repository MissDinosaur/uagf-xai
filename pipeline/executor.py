from layers.explainability.shap_runner import run_shap
from layers.fairness.fairlearn_runner import run_fairness
from layers.uncertainty.mapie_runner import run_uncertainty
from layers.drift.evidently_runner import run_drift
from layers.explainability.dice_runner import run_dice
from layers.explainability.lime_runner import run_lime
from layers.llm.llm_explainability_runner import run_llm_explainability
from layers.llm.llm_fairness_runner import run_llm_fairness
from layers.llm.llm_uncertainty_runner import run_llm_uncertainty
from layers.llm.llm_drift_runner import run_llm_drift
from layers.llm.llm_evidence_methods import (
    LLM_GROUNDING,
    LLM_PROMPT_FAIRNESS,
    LLM_SELF_CONSISTENCY,
    LLM_SEMANTIC_DRIFT,
)
from pipeline.evidence_normalizer import normalize_evidence_results
import time


def _timed_call(producer):
    """Execute one evidence producer and return its result and wall-clock time."""
    started_at = time.perf_counter()
    try:
        result = producer()
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "reason": str(exc),
        }
    return result, time.perf_counter() - started_at


def _attach_runtime_metrics(results, runtimes):
    """Store method timings in the existing unified evidence metrics object."""
    for result_key, runtime_seconds in runtimes.items():
        result = results.get(result_key)
        if not isinstance(result, dict):
            continue
        metrics = result.setdefault("metrics", {})
        if isinstance(metrics, dict):
            metrics["runtime_seconds"] = round(float(runtime_seconds), 6)
    return results


def _normalize_llm_payload(payload):
    """
    Convert golden-set style inputs into the prompt payload expected by the
    current LLM evidence runners.
    """
    if payload is None:
        return {}

    if isinstance(payload, dict) and any(
        key in payload for key in ("current_prompts", "reference_prompts", "fairness_pairs")
    ):
        return payload

    if isinstance(payload, dict) and "golden_set" in payload:
        payload = payload["golden_set"]

    if isinstance(payload, list):
        current_prompts = []
        reference_prompts = []
        fairness_pairs = []

        for item in payload:
            if isinstance(item, dict):
                prompt = item.get("question") or item.get("prompt") or item.get("input") or item.get("query")
                answer = item.get("answer") or item.get("reference") or item.get("output") or item.get("expected_answer")

                if prompt is not None:
                    current_prompts.append(str(prompt))
                if answer is not None:
                    reference_prompts.append(str(answer))
                elif prompt is not None:
                    reference_prompts.append(str(prompt))

                if "group_a_prompt" in item and "group_b_prompt" in item:
                    fairness_pairs.append(
                        {
                            "attribute": item.get("attribute", "unknown"),
                            "group_a_label": item.get("group_a_label", "group_a"),
                            "group_b_label": item.get("group_b_label", "group_b"),
                            "group_a_prompt": item["group_a_prompt"],
                            "group_b_prompt": item["group_b_prompt"],
                        }
                    )
            else:
                prompt = str(item)
                current_prompts.append(prompt)
                reference_prompts.append(prompt)

        return {
            "current_prompts": current_prompts,
            "reference_prompts": reference_prompts,
            "fairness_pairs": fairness_pairs,
            "golden_set": payload,
        }

    if hasattr(payload, "to_dict") and hasattr(payload, "columns"):
        records = payload.to_dict(orient="records")
        return _normalize_llm_payload(records)

    return {
        "current_prompts": [],
        "reference_prompts": [],
        "fairness_pairs": [],
        "golden_set": payload,
    }


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
    system_type="traditional_ml",
    sensitive_features=None,
    provider_name=None,
    output_namespace="audit",
    resource_context=None,
    task_type=None,
    drift_reference_data=None,
    evaluation_frame=None,
    sensitive_data=None,
    feature_columns=None,
    modality=None,
    positive_label=1,
    target_column=None,
    counterfactual_actionable_features=None,
    counterfactual_immutable_features=None,
):

    results = {}
    runtimes = {}
    system_type_normalized = str(system_type).strip().lower()
    is_llm_system = system_type_normalized in {"llm", "agentic"}
    X_raw, X_model, estimator = _resolve_model_views(model, X)

    if is_llm_system:
        llm_payload = _normalize_llm_payload(X_raw)

        if LLM_GROUNDING in methods:
            results[LLM_GROUNDING], runtimes[LLM_GROUNDING] = _timed_call(
                lambda: run_llm_explainability(
                    model,
                    llm_payload,
                    resource_context=resource_context,
                )
            )

        if LLM_SELF_CONSISTENCY in methods:
            results[LLM_SELF_CONSISTENCY], runtimes[LLM_SELF_CONSISTENCY] = _timed_call(
                lambda: run_llm_uncertainty(
                    model,
                    llm_payload,
                    resource_context=resource_context,
                )
            )

        if LLM_SEMANTIC_DRIFT in methods:
            results[LLM_SEMANTIC_DRIFT], runtimes[LLM_SEMANTIC_DRIFT] = _timed_call(
                lambda: run_llm_drift(
                    model,
                    llm_payload,
                    resource_context=resource_context,
                )
            )

        if LLM_PROMPT_FAIRNESS in methods:
            results[LLM_PROMPT_FAIRNESS], runtimes[LLM_PROMPT_FAIRNESS] = _timed_call(
                lambda: run_llm_fairness(
                    model,
                    llm_payload,
                    resource_context=resource_context,
                )
            )

        normalized = normalize_evidence_results(results, task_type=task_type)
        return _attach_runtime_metrics(
            normalized,
            runtimes,
        )

    if "shap" in methods:
        results["explainability"], runtimes["explainability"] = _timed_call(
            lambda: run_shap(
                model,
                X_raw,
                X_model=X_model,
                estimator=estimator,
                provider_name=provider_name,
                output_namespace=output_namespace,
                modality=modality,
            )
        )

    if "lime" in methods:
        results["lime"], runtimes["lime"] = _timed_call(
            lambda: run_lime(
                model,
                X_raw,
                task_type=task_type,
                modality=modality,
                provider_name=provider_name,
                output_namespace=output_namespace,
            )
        )

    if "fairness" in methods:
        if sensitive_features:
            results["fairness"], runtimes["fairness"] = _timed_call(
                lambda: run_fairness(
                    model,
                    X_raw,
                    y,
                    sensitive_data=sensitive_data,
                    sensitive_features=sensitive_features,
                    positive_label=positive_label,
                )
            )
        else:
            results["fairness"], runtimes["fairness"] = _timed_call(
                lambda: {
                    "type": "fairness",
                    "method": "Fairlearn",
                    "status": "skipped",
                    "reason": (
                        "Fairness analysis was skipped because S5 did not provide "
                        "sensitive_feature_columns."
                    ),
                    "note": "No sensitive features specified; fairness check skipped.",
                }
            )

    if "uncertainty" in methods:
        results["uncertainty"], runtimes["uncertainty"] = _timed_call(
            lambda: run_uncertainty(
                model,
                X_raw,
                y,
                task_type=task_type,
                modality=modality,
            )
        )

    if "drift" in methods:
        results["drift"], runtimes["drift"] = _timed_call(
            lambda: run_drift(
                evaluation_frame if evaluation_frame is not None else X_raw,
                reference_data=drift_reference_data,
                model=model,
                modality=modality,
                feature_columns=feature_columns,
                sensitive_feature_columns=sensitive_features,
                target_column=target_column,
                provider_name=provider_name,
                output_namespace=output_namespace,
            )
        )

    if "dice" in methods:
        if hasattr(model, "predict_proba"):
            results["counterfactual"], runtimes["counterfactual"] = _timed_call(
                lambda: run_dice(
                    model,
                    X_raw,
                    y,
                    provider_name=provider_name,
                    output_namespace=output_namespace,
                    sensitive_features=sensitive_features,
                    actionable_features=counterfactual_actionable_features,
                    immutable_features=counterfactual_immutable_features,
                    target_column=target_column,
                )
            )
        else:
            results["counterfactual"], runtimes["counterfactual"] = _timed_call(
                lambda: {
                    "type": "counterfactual",
                    "method": "DiCE",
                    "status": "skipped",
                    "reason": (
                        "DiCE was selected, but the model does not expose the "
                        "predict_proba() interface required by the current runner."
                    ),
                }
            )

    normalized = normalize_evidence_results(results, task_type=task_type)
    return _attach_runtime_metrics(
        normalized,
        runtimes,
    )
