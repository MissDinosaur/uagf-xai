from __future__ import annotations

import pickle
from pathlib import Path

import joblib
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from resources.artifact_utils import resolve_artifact_path
from resources.model_loader import ModelLoader


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("file://data/model.pkl", Path("data/model.pkl")),
        ("relative/model.joblib", Path("relative/model.joblib")),
        ("minio://bucket/path/model.pkl", Path("data/cache/model.pkl")),
    ],
)
def test_artifact_uri_resolution(uri, expected):
    assert resolve_artifact_path(uri) == expected


def test_absolute_path_resolution_is_preserved(tmp_path):
    path = tmp_path / "model.joblib"
    assert resolve_artifact_path(str(path)) == path


def test_single_joblib_model_is_loaded(tmp_path, model_context_factory):
    model = DummyClassifier(strategy="most_frequent").fit([[0], [1]], [0, 1])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)

    loaded = ModelLoader.load(model_context_factory(model_artifact_uri=str(path)))

    assert isinstance(loaded, DummyClassifier)
    assert loaded.predict([[5]]).tolist() == [0]


def test_single_pickle_model_is_loaded(tmp_path, model_context_factory):
    model = DummyRegressor(strategy="mean").fit([[0], [1]], [1.0, 3.0])
    path = tmp_path / "model.pkl"
    with path.open("wb") as handle:
        pickle.dump(model, handle)

    loaded = ModelLoader.load(
        model_context_factory(
            model_artifact_uri=str(path),
            model_format="pkl",
        )
    )

    assert isinstance(loaded, DummyRegressor)
    assert loaded.predict([[8]])[0] == pytest.approx(2.0)


def test_model_directory_loads_entrypoint_and_metadata(tmp_path, model_context_factory):
    directory = tmp_path / "model"
    directory.mkdir()
    model = DummyRegressor(strategy="mean").fit([[0], [1]], [2.0, 4.0])
    joblib.dump(model, directory / "entry.joblib")
    (directory / "deployment_wrapper_meta.json").write_text(
        '{"version": "1.0"}', encoding="utf-8"
    )
    context = model_context_factory(
        model_artifact_uri=str(directory),
        model_format="model_directory",
        model_framework="sklearn",
        model_entrypoint="entry.joblib",
    )

    loaded = ModelLoader.load(context)
    metadata = ModelLoader.load_metadata(context)

    assert isinstance(loaded, DummyRegressor)
    assert metadata["is_directory_contract"] is True
    assert any("deployment_wrapper_meta.json" in item for item in metadata["loaded_metadata_files"])


def test_model_directory_requires_entrypoint(tmp_path, model_context_factory):
    directory = tmp_path / "model"
    directory.mkdir()
    context = model_context_factory(
        model_artifact_uri=str(directory),
        model_format="model_directory",
        model_entrypoint=None,
    )

    with pytest.raises(ValueError, match="model_entrypoint is required"):
        ModelLoader.load(context)


def test_missing_minio_cache_artifact_has_clear_error(model_context_factory):
    uri = "minio://unit-test-bucket/missing-unit-test-model.joblib"
    context = model_context_factory(model_artifact_uri=uri)

    with pytest.raises(FileNotFoundError) as error:
        ModelLoader.load(context)

    assert uri in str(error.value)
    assert "resolved_path" in str(error.value)
