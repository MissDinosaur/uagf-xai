from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from adapters.s5_audit_adapter import AuditContext
from api.audit_api import _load_s5_resources
from resources.loader import ResourceLoader
from resources.resource_bundle import ResourceBundle


def _patch_model_loading(monkeypatch, model=None, metadata=None):
    monkeypatch.setattr(ResourceLoader, "load_model", lambda context: model or object())
    monkeypatch.setattr(ResourceLoader, "load_tokenizer", lambda context: None)
    monkeypatch.setattr(
        ResourceLoader,
        "load_model_metadata",
        lambda context: metadata or {"status": "loaded"},
    )


def test_traditional_contract_loads_training_and_evaluation_csvs(tmp_path, monkeypatch):
    _patch_model_loading(monkeypatch)
    train = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    evaluation = pd.DataFrame({"feature": [3], "target": [1]})
    train_path = tmp_path / "train.csv"
    evaluation_path = tmp_path / "evaluation.csv"
    train.to_csv(train_path, index=False)
    evaluation.to_csv(evaluation_path, index=False)
    context = AuditContext(
        system_type="traditional_ml",
        task_type="binary_classification",
        training_dataset_uri=f"file://{train_path}",
        evaluation_dataset_uri=f"file://{evaluation_path}",
        target_column="target",
    )

    bundle = ResourceLoader.load_bundle(context)

    pd.testing.assert_frame_equal(bundle.training_dataset, train)
    pd.testing.assert_frame_equal(bundle.evaluation_dataset, evaluation)
    assert bundle.golden_dataset is None


def test_llm_contract_loads_golden_set_and_optional_resources(tmp_path, monkeypatch):
    metadata = {"status": "metadata_only", "is_loadable": False}
    model = SimpleNamespace(status="metadata_only", is_loadable=False)
    _patch_model_loading(monkeypatch, model=model, metadata=metadata)
    golden_path = tmp_path / "golden.json"
    prompt_path = tmp_path / "prompt.txt"
    rag_path = tmp_path / "rag.json"
    guardrail_path = tmp_path / "guardrails.json"
    golden_path.write_text(json.dumps([{"prompt": "Hello"}]), encoding="utf-8")
    prompt_path.write_text("Answer carefully.", encoding="utf-8")
    rag_path.write_text(json.dumps({"index": "local"}), encoding="utf-8")
    guardrail_path.write_text(json.dumps({"blocked": []}), encoding="utf-8")
    context = AuditContext(
        system_type="agentic",
        task_type="llm_generation",
        golden_set_uri=f"file://{golden_path}",
        system_prompt_uri=f"file://{prompt_path}",
        rag_manifest_uri=f"file://{rag_path}",
        guardrail_config_uri=f"file://{guardrail_path}",
    )

    bundle = ResourceLoader.load_bundle(context)

    assert bundle.golden_dataset == [{"prompt": "Hello"}]
    assert bundle.system_prompt == "Answer carefully."
    assert bundle.rag_manifest == {"index": "local"}
    assert bundle.guardrail_config == {"blocked": []}
    assert bundle.training_dataset is None
    assert bundle.evaluation_dataset is None


def test_llm_contract_without_golden_set_does_not_require_tabular_data(monkeypatch):
    _patch_model_loading(monkeypatch)
    context = AuditContext(system_type="llm", task_type="llm_generation")

    bundle = ResourceLoader.load_bundle(context)

    assert bundle.golden_dataset is None
    assert bundle.training_dataset is None
    assert bundle.evaluation_dataset is None


def test_system_type_is_authoritative_for_resource_contract_selection():
    context = AuditContext(
        system_type="traditional_ml",
        task_type="llm_generation",
    )

    assert ResourceLoader._is_llm_contract(context) is False


def test_artifact_feature_columns_take_precedence_over_audit_context(monkeypatch):
    training = pd.DataFrame(
        {"artifact_feature": [1], "audit_feature": [2], "target": [0]}
    )
    evaluation = training.copy()
    bundle = ResourceBundle(
        model=object(),
        model_feature_columns=["artifact_feature"],
        model_feature_scope_source="model_artifact_feature_cols",
        training_dataset=training,
        evaluation_dataset=evaluation,
    )
    monkeypatch.setattr(ResourceLoader, "load_bundle", lambda context: bundle)
    context = AuditContext(
        system_type="traditional_ml",
        modality="tabular",
        task_type="binary_classification",
        feature_columns=["audit_feature"],
        target_column="target",
    )

    _, views = _load_s5_resources(context)

    assert views.feature_columns == ["artifact_feature"]
    assert views.feature_scope_source == "model_artifact_feature_cols"
