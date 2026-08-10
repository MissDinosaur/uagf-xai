import pytest

from adapters.s5_audit_adapter import AuditAdapter, AuditContext


def test_audit_context_does_not_expose_output_namespace():
    assert "output_namespace" not in AuditContext.__dataclass_fields__


def test_traditional_stage_b_fields_are_mapped(traditional_s5_json):
    stage_b = traditional_s5_json["client_submission"]["stage_b"]
    stage_b["actionable_feature_columns"] = ["income"]
    stage_b["immutable_feature_columns"] = ["age"]
    context = AuditAdapter.from_audit_report(traditional_s5_json)

    assert context.system_type == "traditional_ml"
    assert context.modality == "tabular"
    assert context.model_artifact_uri == "file://models/model.joblib"
    assert context.model_artifact_kind == "single_file"
    assert context.model_format == "joblib"
    assert context.model_framework == "sklearn"
    assert context.model_type == "binary_classifier"
    assert context.training_dataset_uri == "file://data/train.csv"
    assert context.evaluation_dataset_uri == "file://data/eval.csv"
    assert context.target_column == "target"
    assert context.positive_label == 1
    assert context.sensitive_feature_columns == ["gender"]
    assert context.actionable_feature_columns == ["income"]
    assert context.immutable_feature_columns == ["age"]


def test_old_counterfactual_policy_names_are_not_canonical(traditional_s5_json):
    stage_b = traditional_s5_json["client_submission"]["stage_b"]
    stage_b["counterfactual_actionable_feature_columns"] = ["income"]
    stage_b["counterfactual_immutable_feature_columns"] = ["age"]

    context = AuditAdapter.from_audit_report(traditional_s5_json)

    assert "counterfactual_actionable_feature_columns" not in AuditContext.__dataclass_fields__
    assert "counterfactual_immutable_feature_columns" not in AuditContext.__dataclass_fields__
    assert context.actionable_feature_columns is None
    assert context.immutable_feature_columns == []


def test_duplicate_counterfactual_policy_columns_are_rejected(traditional_s5_json):
    stage_b = traditional_s5_json["client_submission"]["stage_b"]
    stage_b["immutable_feature_columns"] = ["age", "age"]

    with pytest.raises(ValueError, match="must not contain duplicate columns"):
        AuditAdapter.from_audit_report(traditional_s5_json)


def test_nested_stage_b_fields_are_mapped():
    data = {
        "verified_modality": "tabular",
        "verified_risk_tier": "high",
        "cgsa_csp_satisfiable": True,
        "model_artifact_uri": "file://ignored/top-level.joblib",
        "model_artifact_kind": "directory",
        "task_type": "regression",
        "client_submission": {
            "stage_a": {"provider_name": "Nested Provider"},
            "stage_b": {
                "model_artifact_uri": "file://nested/model.joblib",
                "model_artifact_kind": "single_file",
                "model_format": "joblib",
                "model_framework": "sklearn",
                "model_type": "nested_classifier",
                "model_entrypoint": "entry.joblib",
                "training_dataset_uri": "file://nested/train.csv",
                "evaluation_dataset_uri": "file://nested/eval.csv",
                "target_column": "label",
                "positive_label": "yes",
                "sensitive_feature_columns": ["region"],
                "task_type": "binary_classification",
            },
        },
    }

    context = AuditAdapter.from_audit_report(data)

    assert context.system_type == "traditional_ml"
    assert context.modality == "tabular"
    assert context.model_artifact_uri == "file://nested/model.joblib"
    assert context.model_artifact_kind == "single_file"
    assert context.model_entrypoint == "entry.joblib"
    assert context.training_dataset_uri.endswith("train.csv")
    assert context.task_type == "binary_classification"
    assert context.target_column == "label"
    assert context.positive_label == "yes"
    assert context.provider_name == "Nested Provider"


def test_llm_contract_maps_optional_resources_without_tabular_datasets(llm_s5_json):
    context = AuditAdapter.from_audit_report(llm_s5_json)

    assert context.system_type == "agentic"
    assert context.modality == "text"
    assert context.task_type == "llm_generation"
    assert context.golden_set_uri.endswith("golden.json")
    assert context.system_prompt_uri.endswith("system_prompt.txt")
    assert context.rag_manifest_uri.endswith("rag.json")
    assert context.guardrail_config_uri.endswith("guardrails.json")
    assert context.training_dataset_uri is None
    assert context.evaluation_dataset_uri is None
    assert context.target_column is None


def test_is_llm_or_agentic_flag_infers_llm_task():
    context = AuditAdapter.from_audit_report(
        {
            "is_llm_or_agentic": True,
            "model_artifact_uri": "file://model/",
            "model_artifact_kind": "directory",
            "golden_set_uri": "file://golden.json",
            "task_type": "llm_generation",
            "cgsa_csp_satisfiable": True,
        }
    )

    assert context.system_type == "llm"
    assert context.modality == "unknown"
    assert context.task_type == "llm_generation"


@pytest.mark.parametrize("token", ["llm", "rag", "agentic", "mistral", "lora"])
def test_llm_like_model_type_tokens_trigger_llm_inference(token):
    context = AuditAdapter.from_audit_report(
        {
            "is_llm_or_agentic": True,
            "model_type": f"custom_{token}_model",
            "model_artifact_uri": "file://model/",
            "model_artifact_kind": "directory",
            "task_type": "llm_generation",
            "cgsa_csp_satisfiable": True,
        }
    )

    assert context.system_type in {"llm", "agentic"}
    assert context.task_type == "llm_generation"


def test_false_llm_flag_is_authoritative_over_llm_like_metadata():
    context = AuditAdapter.from_audit_report(
        {
            "is_llm_or_agentic": False,
            "system_type": "llm",
            "model_type": "llm_rag_agentic_mistral_lora",
            "modality": "tabular",
            "task_type": "binary_classification",
            "cgsa_csp_satisfiable": True,
        }
    )

    assert context.system_type == "traditional_ml"
    assert context.modality == "tabular"
    assert context.task_type == "binary_classification"


def test_agentic_system_is_derived_from_nested_description_and_tool_calling():
    context = AuditAdapter.from_audit_report(
        {
            "is_llm_or_agentic": True,
            "verified_modality": "llm",
            "client_submission": {
                "stage_a": {"declared_modality": "llm"},
                "stage_b": {
                    "model_type": "foundation_model",
                    "general_description": "An LLM with tool-calling workflows.",
                    "task_type": "llm_generation",
                },
            },
            "cgsa_csp_satisfiable": True,
        }
    )

    assert context.system_type == "agentic"
    assert context.modality == "text"
    assert context.task_type == "llm_generation"


@pytest.mark.parametrize(
    ("raw_modality", "expected"),
    [
        ("tabular", "tabular"),
        ("time-series", "time_series"),
        ("text", "text"),
        ("llm", "text"),
        ("agentic", "text"),
        ("image", "image"),
        ("audio", "audio"),
        ("multimodal", "multimodal"),
        ("unsupported_value", "unknown"),
        (None, "unknown"),
    ],
)
def test_modality_is_normalized_to_the_public_contract(raw_modality, expected):
    context = AuditAdapter.from_audit_report(
        {
            "is_llm_or_agentic": raw_modality in {"llm", "agentic"},
            "modality": raw_modality,
            "cgsa_csp_satisfiable": True,
        }
    )

    assert context.modality == expected


def test_direct_context_normalizes_legacy_values_to_public_contract():
    context = AuditContext(system_type="traditional", modality="time series")

    assert context.system_type == "traditional_ml"
    assert context.modality == "time_series"


def test_invalid_stage_b_model_artifact_kind_is_rejected():
    with pytest.raises(ValueError, match="model_artifact_kind"):
        AuditAdapter.from_audit_report(
            {
                "client_submission": {
                    "stage_b": {"model_artifact_kind": "model_directory"}
                }
            }
        )


def test_talentsift_feature_columns_are_read_from_data_dictionary():
    context = AuditAdapter.from_audit_report(
        {
            "is_llm_or_agentic": False,
            "verified_modality": "nlp",
            "client_submission": {
                "stage_b": {
                    "task_type": "binary_classification",
                    "target_column": "shortlist",
                    "data_dictionary": {"feature_columns": ["cv_text", "cv_text"]},
                }
            },
            "cgsa_csp_satisfiable": True,
        }
    )

    assert context.system_type == "traditional_ml"
    assert context.modality == "text"
    assert context.feature_columns == ["cv_text"]


def test_missing_feature_columns_preserves_none():
    context = AuditAdapter.from_audit_report(
        {
            "client_submission": {
                "stage_b": {"task_type": "binary_classification"}
            },
            "cgsa_csp_satisfiable": True,
        }
    )

    assert context.feature_columns is None


def test_target_column_cannot_be_declared_as_model_feature():
    with pytest.raises(ValueError, match="must not include target_column"):
        AuditAdapter.from_audit_report(
            {
                "client_submission": {
                    "stage_b": {
                        "task_type": "binary_classification",
                        "target_column": "shortlist",
                        "feature_columns": ["cv_text", "shortlist"],
                    }
                }
            }
        )
