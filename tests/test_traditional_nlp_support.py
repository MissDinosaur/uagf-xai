from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adapters.s5_audit_adapter import AuditAdapter
from api.audit_api import _build_evaluation_views
from layers.drift import evidently_runner
from layers.explainability import lime_runner, shap_runner
from layers.fairness.fairlearn_runner import run_fairness
from layers.uncertainty.mapie_runner import _model_view, run_uncertainty
from resources.model_loader import ModelLoader
from resources.traditional_model_loader import SklearnTextModelAdapter


CASE_DIR = Path("data/05_talentsift_gmbh")


def _talentsift_context():
    path = CASE_DIR / "s5_talentsift_gmbh_audit_state.json"
    return AuditAdapter.from_audit_report(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture
def talentsift_model():
    return ModelLoader.load(_talentsift_context())


def test_evaluation_views_keep_model_fairness_and_audit_columns_separate():
    context = _talentsift_context()
    evaluation = pd.read_csv(CASE_DIR / "datasets/evaluation_dataset.csv")

    views = _build_evaluation_views(
        evaluation,
        target_column=context.target_column,
        feature_columns=context.feature_columns,
        sensitive_feature_columns=context.sensitive_feature_columns,
        modality=context.modality,
    )

    assert list(views.X_model.columns) == ["cv_text"]
    assert views.y.name == "shortlist"
    assert list(views.sensitive_data.columns) == ["sex", "age_group", "nationality"]
    assert "applicant_id" not in views.X_model
    assert "vacancy_family" not in views.X_model
    assert views.X_model.index.equals(views.y.index)
    assert views.sensitive_data.index.equals(views.y.index)

    missing = pd.DataFrame({"shortlist": [0, 1], "other": ["a", "b"]})
    with pytest.raises(ValueError, match="model feature columns are missing"):
        _build_evaluation_views(
            missing,
            target_column="shortlist",
            feature_columns=["cv_text"],
            sensitive_feature_columns=[],
            modality="text",
        )

    legacy = pd.DataFrame({"feature": [1], "audit_column": [2], "target": [0]})
    legacy_views = _build_evaluation_views(
        legacy,
        target_column="target",
        feature_columns=None,
        sensitive_feature_columns=[],
        modality="tabular",
    )
    assert list(legacy_views.X_model.columns) == ["feature", "audit_column"]


def test_talentsift_artifact_and_adapter_delegate_to_fitted_pipeline(
    talentsift_model,
    monkeypatch,
):
    model = talentsift_model
    evaluation = pd.read_csv(CASE_DIR / "datasets/evaluation_dataset.csv")
    X_model = evaluation[["cv_text"]].head(3)

    assert isinstance(model, SklearnTextModelAdapter)
    assert model.text_feature_column == "cv_text"
    assert model.vectorizer_class == "TfidfVectorizer"
    assert model.estimator_class == "LogisticRegression"
    assert model.vocabulary_size == 742
    assert not any(name.startswith("feature_") for name in model.get_feature_names_out())
    pipeline = model.underlying_pipeline
    vectorizer = model.vectorizer
    original_predict = pipeline.predict
    original_predict_proba = pipeline.predict_proba
    original_transform = vectorizer.transform
    calls = {"predict": [], "predict_proba": [], "transform": []}

    def predict_spy(frame):
        calls["predict"].append(frame.copy())
        return original_predict(frame)

    def predict_proba_spy(frame):
        calls["predict_proba"].append(frame.copy())
        return original_predict_proba(frame)

    def transform_spy(raw_texts):
        calls["transform"].append(list(raw_texts))
        return original_transform(raw_texts)

    monkeypatch.setattr(pipeline, "predict", predict_spy)
    monkeypatch.setattr(pipeline, "predict_proba", predict_proba_spy)
    monkeypatch.setattr(vectorizer, "transform", transform_spy)

    prepared = model.prepare_input(X_model)
    predictions = model.predict(X_model)
    probabilities = model.predict_proba(X_model["cv_text"].tolist())

    assert prepared.shape == (3, 742)
    assert predictions.shape == (3,)
    assert probabilities.shape == (3, 2)
    assert len(calls["predict"]) == 1
    assert len(calls["predict_proba"]) == 1
    assert len(calls["transform"]) == 3
    assert all(len(values) == 3 for values in calls["transform"])
    assert list(calls["predict"][0].columns) == ["cv_text"]
    assert list(calls["predict_proba"][0].columns) == ["cv_text"]


def test_text_adapter_never_calls_training_or_clone_operations(
    talentsift_model,
    monkeypatch,
):
    evaluation = pd.read_csv(CASE_DIR / "datasets/evaluation_dataset.csv")
    X_model = evaluation[["cv_text"]].head(4)
    estimator = talentsift_model.estimator
    coefficients = estimator.coef_.copy()
    intercept = estimator.intercept_.copy()
    classes = estimator.classes_.copy()

    def reject_training(*args, **kwargs):
        raise AssertionError("The fitted S5 artifact must never be trained or cloned.")

    with monkeypatch.context() as patch:
        objects_and_methods = (
            (talentsift_model.underlying_pipeline, "fit"),
            (talentsift_model.vectorizer, "fit"),
            (talentsift_model.vectorizer, "fit_transform"),
            (talentsift_model.estimator, "fit"),
            (talentsift_model.estimator, "partial_fit"),
        )
        for target, method_name in objects_and_methods:
            if hasattr(target, method_name):
                patch.setattr(target, method_name, reject_training)
        patch.setattr("resources.traditional_model_loader.clone", reject_training)

        talentsift_model.prepare_input(X_model)
        talentsift_model.predict(X_model)
        talentsift_model.predict_proba(X_model)
        talentsift_model.decision_function(X_model)

    np.testing.assert_array_equal(estimator.coef_, coefficients)
    np.testing.assert_array_equal(estimator.intercept_, intercept)
    np.testing.assert_array_equal(estimator.classes_, classes)


def test_text_shap_reports_fitted_vocabulary_tokens(
    talentsift_model,
    monkeypatch,
    tmp_path,
):
    evaluation = pd.read_csv(CASE_DIR / "datasets/evaluation_dataset.csv")
    X_model = evaluation[["cv_text"]].head(4)
    output_path = tmp_path / "tokens.png"
    monkeypatch.setattr(
        shap_runner,
        "build_output_path",
        lambda *args, **kwargs: output_path,
    )

    result = shap_runner.run_shap(
        talentsift_model,
        X_model,
        modality="text",
        provider_name="test",
    )

    assert result["status"] == "completed"
    assert result["explainer"] == "LinearExplainer"
    assert result["feature_semantics"] == "tokens_and_ngrams"
    assert result["global_token_importance"]
    assert not result["top_features"][0].startswith("feature_")
    assert output_path.exists()


def test_text_lime_uses_raw_text_prediction_callback(
    talentsift_model,
    monkeypatch,
    tmp_path,
):
    evaluation = pd.read_csv(CASE_DIR / "datasets/evaluation_dataset.csv")
    X_model = evaluation[["cv_text"]].head(4)
    output_path = tmp_path / "lime.html"
    monkeypatch.setattr(
        lime_runner,
        "build_output_path",
        lambda *args, **kwargs: output_path,
    )
    monkeypatch.setattr(
        lime_runner,
        "_numeric_frame",
        lambda value: pytest.fail("Text LIME must not use the numeric frame route."),
    )

    result = lime_runner.run_lime(
        talentsift_model,
        X_model,
        task_type="binary_classification",
        modality="text",
        provider_name="test",
    )

    assert result["status"] == "completed"
    assert result["explainer"] == "LimeTextExplainer"
    assert result["input_representation"] == "raw_text"
    assert result["token_contributions"]
    assert output_path.exists()


def test_text_mapie_uses_transform_without_refitting(talentsift_model, monkeypatch):
    evaluation = pd.read_csv(CASE_DIR / "datasets/evaluation_dataset.csv")
    X_model = evaluation[["cv_text"]]
    y = evaluation["shortlist"]
    estimator = talentsift_model.estimator
    selected_estimator, prepared = _model_view(talentsift_model, X_model)
    original_predict = estimator.predict
    original_predict_proba = estimator.predict_proba
    calls = {"predict": 0, "predict_proba": 0}

    def reject_fit(*args, **kwargs):
        raise AssertionError("The fitted S5 text artifact must not be retrained.")

    def predict_spy(values):
        calls["predict"] += 1
        return original_predict(values)

    def predict_proba_spy(values):
        calls["predict_proba"] += 1
        return original_predict_proba(values)

    monkeypatch.setattr(talentsift_model.vectorizer, "fit", reject_fit)
    monkeypatch.setattr(talentsift_model.vectorizer, "fit_transform", reject_fit)
    monkeypatch.setattr(estimator, "fit", reject_fit)
    monkeypatch.setattr(estimator, "predict", predict_spy)
    monkeypatch.setattr(estimator, "predict_proba", predict_proba_spy)

    result = run_uncertainty(
        talentsift_model,
        X_model,
        y,
        task_type="binary_classification",
        modality="text",
    )

    assert result["status"] == "completed"
    assert result["input_representation"] == "tfidf_dense_for_mapie"
    assert result["estimator_class"] == "LogisticRegression"
    assert 0.0 <= result["coverage"] <= 1.0
    assert selected_estimator is estimator
    assert prepared.shape == (80, 742)
    assert calls["predict_proba"] > 0
    assert calls["predict"] > 0


def test_text_drift_uses_only_declared_text_feature(
    talentsift_model,
    monkeypatch,
    tmp_path,
):
    reference = pd.read_csv(CASE_DIR / "datasets/training_dataset.csv").head(30)
    current = pd.read_csv(CASE_DIR / "datasets/evaluation_dataset.csv").head(30)
    output_path = tmp_path / "text_drift.json"
    monkeypatch.setattr(
        evidently_runner,
        "_run_evidently",
        lambda reference_frame, current_frame: ({"engine": "test"}, None),
    )
    monkeypatch.setattr(
        evidently_runner,
        "_save_text_drift",
        lambda payload, provider_name, output_namespace: str(output_path),
    )

    result = evidently_runner.run_drift(
        current,
        reference_data=reference,
        model=talentsift_model,
        modality="text",
        feature_columns=["cv_text"],
        sensitive_feature_columns=["sex", "age_group", "nationality"],
        target_column="shortlist",
    )

    assert result["status"] == "completed"
    assert result["metrics"]["feature_columns"] == ["cv_text"]
    derived = {item["feature"] for item in result["metrics"]["derived_feature_tests"]}
    assert derived == {
        "character_count",
        "word_count",
        "unique_token_ratio",
        "digit_ratio",
    }


def test_fairness_predictions_receive_model_features_only():
    class ModelSpy:
        def predict(self, X):
            assert list(X.columns) == ["cv_text"]
            return [1, 0, 1, 0, 1, 0]

    X_model = pd.DataFrame({"cv_text": ["text"] * 6})
    y = pd.Series([1, 0, 1, 0, 0, 1])
    sensitive = pd.DataFrame({"sex": ["f", "m", "f", "m", "f", "m"]})

    result = run_fairness(
        ModelSpy(),
        X_model,
        y,
        sensitive_data=sensitive,
        sensitive_features=["sex"],
        positive_label=1,
    )

    assert result["status"] == "completed"
    assert "demographic_parity_difference" in result["sex"]
