from __future__ import annotations

import joblib
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from resources.llm_model_loader import MetadataOnlyLLMArtifact
from resources.model_loader import (
    EncodedSklearnModel as FacadeEncodedSklearnModel,
    MetadataOnlyLLMArtifact as FacadeMetadataOnlyLLMArtifact,
    ModelLoader,
)
from resources.traditional_model_loader import EncodedSklearnModel


@pytest.fixture
def encoded_bundle(tmp_path, model_context_factory):
    encoder = LabelEncoder().fit(["blue", "red"])
    X_model = pd.DataFrame({"color": [0, 1, 0, 1], "value": [0, 0, 1, 1]})
    estimator = DecisionTreeClassifier(random_state=7).fit(X_model, [0, 1, 0, 1])
    path = tmp_path / "bundle.joblib"
    joblib.dump(
        {
            "model": estimator,
            "encoders": {"color": encoder},
            "feature_cols": ["color", "value"],
        },
        path,
    )
    context = model_context_factory(model_artifact_uri=str(path))
    return ModelLoader.load(context)


def test_sklearn_bundle_is_wrapped_and_predicts_raw_data(encoded_bundle):
    raw = pd.DataFrame({"value": [0, 1], "color": ["blue", "red"]})

    prepared = encoded_bundle.prepare_input(raw)
    predictions = encoded_bundle.predict(raw)

    assert isinstance(encoded_bundle, EncodedSklearnModel)
    assert list(prepared.columns) == ["color", "value"]
    assert prepared["color"].tolist() == [0, 1]
    assert predictions.tolist() == [0, 1]
    assert encoded_bundle.estimator is encoded_bundle.model


def test_model_loader_facade_preserves_legacy_wrapper_exports():
    assert FacadeEncodedSklearnModel is EncodedSklearnModel
    assert FacadeMetadataOnlyLLMArtifact is MetadataOnlyLLMArtifact


def test_sklearn_bundle_exposes_predict_proba(encoded_bundle):
    raw = pd.DataFrame({"color": ["blue"], "value": [0]})
    probabilities = encoded_bundle.predict_proba(raw)

    assert probabilities.shape == (1, 2)


def test_unseen_bundle_category_raises_clear_error(encoded_bundle):
    raw = pd.DataFrame({"color": ["green"], "value": [0]})

    with pytest.raises(ValueError, match="unseen categories") as error:
        encoded_bundle.prepare_input(raw)

    assert "green" in str(error.value)


def test_missing_bundle_feature_raises_clear_error(encoded_bundle):
    with pytest.raises(ValueError, match="missing required model features"):
        encoded_bundle.prepare_input(pd.DataFrame({"color": ["blue"]}))


def test_metadata_only_huggingface_folder_returns_structured_artifact(
    tmp_path, model_context_factory
):
    directory = tmp_path / "model"
    source = directory / "stub"
    source.mkdir(parents=True)
    for filename in (
        "config.json",
        "adapter_config.json",
        "generation_config.json",
        "tokenizer_config.json",
    ):
        (source / filename).write_text("{}", encoding="utf-8")
    context = model_context_factory(
        model_artifact_uri=str(directory),
        model_format="model_directory",
        model_framework="huggingface",
        model_type="llm_rag_model",
        model_entrypoint="stub",
        system_type="agentic",
    )

    artifact = ModelLoader.load(context)
    metadata = ModelLoader.load_metadata(context)

    assert isinstance(artifact, MetadataOnlyLLMArtifact)
    assert artifact.status == "metadata_only"
    assert artifact.is_loadable is False
    assert "config.json" in artifact.loaded_metadata_files
    assert metadata["status"] == "metadata_only"
    assert metadata["is_loadable"] is False
