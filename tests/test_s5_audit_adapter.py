import pytest

from adapters.s5_audit_adapter import AuditAdapter


def test_flat_traditional_fields_are_mapped(traditional_s5_json):
    context = AuditAdapter.from_audit_report(traditional_s5_json)

    assert context.model_artifact_uri == "file://models/model.joblib"
    assert context.model_format == "joblib"
    assert context.model_framework == "sklearn"
    assert context.model_type == "binary_classifier"
    assert context.training_dataset_uri == "file://data/train.csv"
    assert context.evaluation_dataset_uri == "file://data/eval.csv"
    assert context.target_column == "target"
    assert context.positive_label == 1
    assert context.sensitive_feature_columns == ["gender"]


def test_nested_stage_b_fields_are_mapped():
    data = {
        "verified_modality": "tabular",
        "verified_risk_tier": "high",
        "cgsa_csp_satisfiable": True,
        "client_submission": {
            "stage_a": {"provider_name": "Nested Provider"},
            "stage_b": {
                "model_artifact_uri": "file://nested/model.joblib",
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

    assert context.model_artifact_uri == "file://nested/model.joblib"
    assert context.model_entrypoint == "entry.joblib"
    assert context.training_dataset_uri.endswith("train.csv")
    assert context.target_column == "label"
    assert context.positive_label == "yes"
    assert context.provider_name == "Nested Provider"


def test_llm_contract_maps_optional_resources_without_tabular_datasets(llm_s5_json):
    context = AuditAdapter.from_audit_report(llm_s5_json)

    assert context.system_type == "agentic"
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
            "golden_set_uri": "file://golden.json",
            "cgsa_csp_satisfiable": True,
        }
    )

    assert context.system_type == "llm"
    assert context.task_type == "llm_generation"


@pytest.mark.parametrize("token", ["llm", "rag", "agentic", "mistral", "lora"])
def test_llm_like_model_type_tokens_trigger_llm_inference(token):
    context = AuditAdapter.from_audit_report(
        {
            "model_type": f"custom_{token}_model",
            "model_artifact_uri": "file://model/",
            "cgsa_csp_satisfiable": True,
        }
    )

    assert context.system_type in {"llm", "agentic"}
    assert context.task_type == "llm_generation"
