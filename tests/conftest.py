from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from adapters.s4_governance_adapter import GovernanceContext
from adapters.s5_audit_adapter import AuditContext
from schema.evidence_schema import completed_evidence


@pytest.fixture
def minimal_governance_json():
    return {
        "overall_scores": {
            "composite_maturity_score": 3.4,
            "governance_verdict": "PASS_WITH_OBSERVATIONS",
        },
        "domains": [
            {"domain_name": "Transparency and Explainability", "domain_score": 3.1},
            {"domain_name": "Monitoring and Incident Response", "domain_score": 2.9},
        ],
    }


@pytest.fixture
def governance_context():
    return GovernanceContext(
        governance_score=3.4,
        governance_verdict="PASS_WITH_OBSERVATIONS",
        domain_scores={
            "Transparency and Explainability": 3.1,
            "Monitoring and Incident Response": 2.9,
        },
    )


@pytest.fixture
def traditional_s5_json():
    return {
        "system_type": "traditional",
        "modality": "tabular",
        "risk_tier": "high",
        "model_artifact_uri": "file://models/model.joblib",
        "model_format": "joblib",
        "model_framework": "sklearn",
        "model_type": "binary_classifier",
        "model_entrypoint": None,
        "training_dataset_uri": "file://data/train.csv",
        "evaluation_dataset_uri": "file://data/eval.csv",
        "target_column": "target",
        "positive_label": 1,
        "sensitive_feature_columns": ["gender"],
        "task_type": "binary_classification",
        "provider_name": "Test Provider",
        "cgsa_csp_satisfiable": True,
    }


@pytest.fixture
def llm_s5_json():
    return {
        "system_type": "agentic",
        "modality": "text",
        "risk_tier": "high",
        "model_artifact_uri": "file://models/llm/",
        "model_format": "model_directory",
        "model_framework": "huggingface",
        "model_type": "llm_rag_agentic_mistral_lora",
        "model_entrypoint": "stub",
        "golden_set_uri": "file://data/golden.json",
        "system_prompt_uri": "file://data/system_prompt.txt",
        "rag_manifest_uri": "file://data/rag.json",
        "guardrail_config_uri": "file://data/guardrails.json",
        "cgsa_csp_satisfiable": True,
    }


@pytest.fixture
def small_mixed_frame():
    return pd.DataFrame(
        {
            "age": [22, 35, 47, 52],
            "income": [30_000, 50_000, 70_000, 90_000],
            "gender": ["female", "male", "female", "male"],
            "target": [0, 1, 1, 0],
        }
    )


@pytest.fixture
def traditional_audit_context():
    return AuditContext(
        system_type="traditional",
        modality="tabular",
        application_domain="finance",
        risk_tier="high",
        applicable_articles=["Art10", "Art13", "Art15"],
        sensitive_feature_columns=["gender"],
        target_column="target",
        positive_label=1,
        task_type="binary_classification",
        provider_name="Test Provider",
    )


@pytest.fixture
def unified_result_factory():
    def factory(
        evidence_id="EXP-SHAP",
        layer="explainability",
        method="SHAP",
        **overrides,
    ):
        payload = {
            "evidence_id": evidence_id,
            "layer": layer,
            "method": method,
            "article_mapping": ["Art. 13"],
            "summary": "Synthetic evidence summary.",
            "key_findings": ["Synthetic finding."],
            "metrics": {"score": 0.8},
            "artifacts": [],
            "limitations": ["Synthetic limitation."],
            "raw_output": {"marker": "raw-marker"},
        }
        payload.update(overrides)
        return completed_evidence(**payload)

    return factory


@pytest.fixture
def model_context_factory():
    def factory(**overrides):
        values = {
            "model_artifact_uri": None,
            "model_format": "joblib",
            "model_framework": "sklearn",
            "model_type": "test_model",
            "model_entrypoint": None,
            "system_type": "traditional",
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    return factory
