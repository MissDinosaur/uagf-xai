import pytest

from layers.llm.llm_evidence_methods import LLM_GROUNDING
from pipeline.evidence_normalizer import REQUIRED_FIELDS, normalize_evidence_results
from schema.evidence_schema import EVIDENCE_STATUSES, completed_evidence


@pytest.mark.parametrize(
    ("result_key", "raw"),
    [
        (
            "explainability",
            {
                "type": "explainability",
                "method": "SHAP",
                "top_features": ["age"],
                "feature_importance": [{"feature": "age", "importance": 0.7}],
            },
        ),
        (
            "fairness",
            {
                "type": "fairness",
                "method": "Fairlearn",
                "gender": {
                    "demographic_parity_difference": 0.1,
                    "equalized_odds_difference": 0.05,
                },
            },
        ),
        (
            "uncertainty",
            {
                "type": "uncertainty",
                "method": "MAPIE",
                "confidence_level": 0.9,
                "coverage": 0.88,
                "mean_interval_width": 1.2,
                "coverage_gap": 0.02,
            },
        ),
        (
            "drift",
            {
                "type": "drift",
                "method": "Evidently",
                "drift_share": 0.25,
                "features_analyzed": 4,
                "drifted_features": ["age"],
            },
        ),
        (
            "counterfactual",
            {
                "type": "counterfactual",
                "method": "DiCE",
                "counterfactuals_count": 2,
                "output": "counterfactuals.json",
            },
        ),
    ],
)
def test_raw_runner_outputs_are_normalized_and_preserved(result_key, raw):
    result = normalize_evidence_results({result_key: raw})[result_key]

    assert REQUIRED_FIELDS <= set(result)
    assert result["status"] == "completed"
    assert result["raw_output"] == raw


def test_llm_skipped_output_becomes_unified_evidence():
    raw = {
        "type": "llm_evidence",
        "method": "LLM-E1 Grounding Score",
        "status": "skipped",
        "reason": "metadata-only",
        "available_resources": ["golden_set", "model_metadata"],
    }

    result = normalize_evidence_results({LLM_GROUNDING: raw})[LLM_GROUNDING]

    assert result["evidence_id"] == "LLM-E1"
    assert result["status"] == "skipped"
    assert result["raw_output"] == raw


def test_llm_failed_output_remains_failed():
    raw = {
        "status": "failed",
        "error_type": "RuntimeError",
        "reason": "generation returned no text",
    }

    result = normalize_evidence_results({LLM_GROUNDING: raw})[LLM_GROUNDING]

    assert result["status"] == "failed"
    assert "generation returned no text" in result["summary"]


def test_existing_unified_evidence_is_preserved():
    unified = completed_evidence(
        evidence_id="CUSTOM",
        layer="explainability",
        method="SHAP",
        metrics={"score": 1.0},
        raw_output={"source": "already-unified"},
    )

    result = normalize_evidence_results({"explainability": unified})

    assert result["explainability"] == unified


def test_drift_normalization_preserves_canonical_detection_fields():
    raw = {
        "method": "Evidently",
        "canonical_source": "evidently",
        "dataset_drift_detected": True,
        "drift_share": 0.4,
        "drifted_feature_count": 2,
        "features_analyzed": 5,
        "drifted_features": ["a", "b"],
        "uagf_dataset_drift_share_threshold": 0.2,
    }

    result = normalize_evidence_results({"drift": raw})["drift"]

    assert result["metrics"]["canonical_source"] == "evidently"
    assert result["metrics"]["dataset_drift_detected"] is True
    assert result["metrics"]["drifted_feature_count"] == 2
    assert result["metrics"]["uagf_dataset_drift_share_threshold"] == 0.2


def test_incompatible_methods_receive_not_applicable_results():
    assessment = {
        "task_type": "forecasting",
        "incompatible_methods": ["fairness", "dice"],
    }
    trace = {"task_type": "forecasting", "task_compatibility_assessment": assessment}

    result = normalize_evidence_results({}, cbep_trace=trace, task_type="forecasting")

    assert result["fairness"]["status"] == "not_applicable"
    assert result["counterfactual"]["status"] == "not_applicable"


def test_selected_method_without_executor_output_is_failed():
    trace = {"task_type": "binary_classification", "final_plan": ["shap"]}

    result = normalize_evidence_results({}, cbep_trace=trace)

    assert result["explainability"]["status"] == "failed"
    assert "executor returned no evidence" in result["explainability"]["summary"]


def test_all_normalized_statuses_are_allowed():
    raw_results = {
        "explainability": {"status": "skipped", "reason": "not available"},
        "fairness": {"status": "failed", "error": "metric error"},
    }

    normalized = normalize_evidence_results(raw_results)

    assert all(item["status"] in EVIDENCE_STATUSES for item in normalized.values())


def test_structured_dice_output_is_normalized_as_explainability():
    raw = {
        "type": "explainability",
        "method": "DiCE",
        "evidence_type": "counterfactual_explanation",
        "original_prediction": 1,
        "original_prediction_proba": [0.1, 0.9],
        "desired_class": "opposite",
        "counterfactuals_count": 1,
        "counterfactual_policy_source": "audit_context_immutable_exclusions",
        "counterfactual_policy_status": "validated",
        "actionable_feature_columns": None,
        "immutable_feature_columns": ["age", "credit_history"],
        "excluded_sensitive_features": ["personal_status"],
        "excluded_immutable_features": ["age", "credit_history"],
        "excluded_non_actionable_features": [],
        "features_to_vary": ["income"],
        "policy_violation_detected": False,
        "counterfactuals": [
            {
                "counterfactual_id": 1,
                "counterfactual_prediction": 0,
                "counterfactual_prediction_proba": [0.8, 0.2],
                "changed_features": [
                    {
                        "feature": "income",
                        "original_value": 80_000,
                        "counterfactual_value": 40_000,
                        "delta": -40_000,
                    }
                ],
                "number_of_changed_features": 1,
            }
        ],
        "output": "outputs/dice/test_counterfactuals.json",
        "limitations": ["Domain review is required."],
    }

    result = normalize_evidence_results({"counterfactual": raw})["counterfactual"]

    assert result["evidence_id"] == "EXP-DICE"
    assert result["layer"] == "explainability"
    assert result["method"] == "DiCE"
    assert result["metrics"]["changed_features_count"] == 1
    assert result["metrics"]["changed_features"][0]["feature"] == "income"
    assert result["metrics"]["counterfactual_policy_status"] == "validated"
    assert result["metrics"]["excluded_immutable_features"] == [
        "age",
        "credit_history",
    ]
    assert result["metrics"]["features_to_vary"] == ["income"]
    assert result["metrics"]["policy_violation_detected"] is False
    assert result["raw_output"] == raw


def test_forecasting_uncertainty_is_preserved_in_common_schema():
    raw = {
        "type": "uncertainty",
        "method": "MAPIE (Conformal Prediction)",
        "status": "completed",
        "task_type": "forecasting",
        "prediction_interval_kind": "numeric_prediction_interval",
        "estimator_mode": "prefit",
        "calibration_policy": "first_half_of_evaluation_data",
        "calibration_rows": 20,
        "measurement_rows": 20,
        "confidence_level": 0.9,
        "coverage": 0.85,
        "mean_interval_width": 2.4,
        "coverage_gap": 0.05,
    }

    result = normalize_evidence_results({"uncertainty": raw})["uncertainty"]

    assert REQUIRED_FIELDS <= set(result)
    assert result["status"] == "completed"
    assert result["metrics"]["task_type"] == "forecasting"
    assert result["metrics"]["prediction_interval_kind"] == "numeric_prediction_interval"
    assert "temporal dependence" in " ".join(result["limitations"])
