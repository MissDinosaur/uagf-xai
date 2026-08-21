import pytest

from adapters.s5_audit_adapter import AuditAdapter, AuditContext


def test_audit_context_excludes_internal_and_legacy_fields(traditional_s5_json):
    excluded = {
        "output_namespace",
        "evaluation_embedding_model",
        "evaluation_embedding_model_uri",
        "semantic_drift_dataset_uri",
        "fairness_prompt_pairs_uri",
        "validation_context_origin",
        "governance_scenario_origin",
        "counterfactual_actionable_feature_columns",
        "counterfactual_immutable_feature_columns",
    }
    stage_b = traditional_s5_json["client_submission"]["stage_b"]
    for field_name in excluded:
        stage_b[field_name] = "must_be_ignored"

    context = AuditAdapter.from_audit_report(traditional_s5_json)

    assert excluded.isdisjoint(AuditContext.__dataclass_fields__)
    assert all(not hasattr(context, field_name) for field_name in excluded)
    assert context.actionable_feature_columns is None
    assert context.immutable_feature_columns == []


def test_traditional_contract_maps_stage_b_and_prefers_nested_values(
    traditional_s5_json,
):
    stage_b = traditional_s5_json["client_submission"]["stage_b"]
    stage_b.update(
        {
            "model_artifact_uri": "file://nested/model.joblib",
            "model_entrypoint": "entry.joblib",
            "actionable_feature_columns": ["income"],
            "immutable_feature_columns": ["age"],
        }
    )
    traditional_s5_json.update(
        {
            "model_artifact_uri": "file://ignored/top-level.joblib",
            "model_artifact_kind": "directory",
            "task_type": "regression",
        }
    )

    context = AuditAdapter.from_audit_report(traditional_s5_json)

    assert context.system_type == "traditional_ml"
    assert context.modality == "tabular"
    assert context.model_artifact_uri == "file://nested/model.joblib"
    assert context.model_artifact_kind == "single_file"
    assert context.model_entrypoint == "entry.joblib"
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
    assert context.task_type == "binary_classification"
    assert context.provider_name == "Test Provider"


def test_counterfactual_policy_rejects_duplicate_columns(traditional_s5_json):
    stage_b = traditional_s5_json["client_submission"]["stage_b"]
    stage_b["immutable_feature_columns"] = ["age", "age"]

    with pytest.raises(ValueError, match="must not contain duplicate columns"):
        AuditAdapter.from_audit_report(traditional_s5_json)


def test_llm_contract_maps_optional_resources_without_tabular_data(llm_s5_json):
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


def test_system_type_derivation_covers_traditional_llm_and_agentic_cases():
    cases = [
        (
            {
                "is_llm_or_agentic": False,
                "system_type": "llm",
                "model_type": "llm_rag_agentic_mistral_lora",
                "modality": "tabular",
                "task_type": "binary_classification",
                "cgsa_csp_satisfiable": True,
            },
            ("traditional_ml", "binary_classification"),
        ),
        (
            {
                "is_llm_or_agentic": True,
                "model_artifact_kind": "directory",
                "task_type": "llm_generation",
                "cgsa_csp_satisfiable": True,
            },
            ("llm", "llm_generation"),
        ),
        (
            {
                "is_llm_or_agentic": True,
                "verified_modality": "llm",
                "client_submission": {
                    "stage_b": {
                        "model_type": "foundation_model",
                        "general_description": "An LLM with tool-calling workflows.",
                        "task_type": "llm_generation",
                    }
                },
                "cgsa_csp_satisfiable": True,
            },
            ("agentic", "llm_generation"),
        ),
    ]
    for payload, expected in cases:
        context = AuditAdapter.from_audit_report(payload)
        assert (context.system_type, context.task_type) == expected

    for token in ("llm", "rag", "agentic", "mistral", "lora"):
        context = AuditAdapter.from_audit_report(
            {
                "is_llm_or_agentic": True,
                "client_submission": {
                    "stage_b": {
                        "model_type": f"custom_{token}_model",
                        "task_type": "llm_generation",
                    }
                },
                "cgsa_csp_satisfiable": True,
            }
        )
        assert context.system_type in {"llm", "agentic"}


def test_modality_normalization_covers_the_public_contract():
    cases = {
        "tabular": "tabular",
        "time-series": "time_series",
        "text": "text",
        "llm": "text",
        "agentic": "text",
        "nlp": "text",
        "image": "image",
        "audio": "audio",
        "multimodal": "multimodal",
        "unsupported_value": "unknown",
        None: "unknown",
    }
    for raw_modality, expected in cases.items():
        context = AuditAdapter.from_audit_report(
            {
                "is_llm_or_agentic": raw_modality in {"llm", "agentic"},
                "modality": raw_modality,
                "cgsa_csp_satisfiable": True,
            }
        )
        assert context.modality == expected

    direct = AuditContext(system_type="traditional", modality="time series")
    assert direct.system_type == "traditional_ml"
    assert direct.modality == "time_series"


def test_invalid_model_artifact_kind_is_rejected():
    with pytest.raises(ValueError, match="model_artifact_kind"):
        AuditAdapter.from_audit_report(
            {
                "client_submission": {
                    "stage_b": {"model_artifact_kind": "model_directory"}
                }
            }
        )


def test_feature_columns_are_normalized_and_validated():
    base = {
        "is_llm_or_agentic": False,
        "verified_modality": "nlp",
        "cgsa_csp_satisfiable": True,
        "client_submission": {
            "stage_b": {
                "task_type": "binary_classification",
                "target_column": "shortlist",
                "data_dictionary": {"feature_columns": ["cv_text", "cv_text"]},
            }
        },
    }
    context = AuditAdapter.from_audit_report(base)
    assert context.feature_columns == ["cv_text"]
    assert context.modality == "text"

    missing = AuditAdapter.from_audit_report(
        {
            "client_submission": {
                "stage_b": {"task_type": "binary_classification"}
            },
            "cgsa_csp_satisfiable": True,
        }
    )
    assert missing.feature_columns is None

    base["client_submission"]["stage_b"]["feature_columns"] = [
        "cv_text",
        "shortlist",
    ]
    with pytest.raises(ValueError, match="must not include target_column"):
        AuditAdapter.from_audit_report(base)
