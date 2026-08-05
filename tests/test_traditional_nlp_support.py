from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from adapters.s5_audit_adapter import AuditAdapter
from api.audit_api import _build_evaluation_views
from layers.drift import evidently_runner
from layers.explainability import lime_runner, shap_runner
from layers.fairness.fairlearn_runner import run_fairness
from layers.uncertainty.mapie_runner import run_uncertainty
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


def test_evaluation_views_reject_missing_declared_model_feature():
    evaluation = pd.DataFrame({"shortlist": [0, 1], "other": ["a", "b"]})

    with pytest.raises(ValueError, match="model feature columns are missing"):
        _build_evaluation_views(
            evaluation,
            target_column="shortlist",
            feature_columns=["cv_text"],
            sensitive_feature_columns=[],
            modality="text",
        )


def test_legacy_evaluation_view_preserves_target_drop_fallback():
    evaluation = pd.DataFrame({"feature": [1], "audit_column": [2], "target": [0]})

    views = _build_evaluation_views(
        evaluation,
        target_column="target",
        feature_columns=None,
        sensitive_feature_columns=[],
        modality="tabular",
    )

    assert list(views.X_model.columns) == ["feature", "audit_column"]


def test_talentsift_artifact_loads_as_fitted_text_adapter(talentsift_model):
    model = talentsift_model
    evaluation = pd.read_csv(CASE_DIR / "datasets/evaluation_dataset.csv")
    X_model = evaluation[["cv_text"]].head(3)

    assert isinstance(model, SklearnTextModelAdapter)
    assert model.text_feature_column == "cv_text"
    assert model.vectorizer_class == "TfidfVectorizer"
    assert model.estimator_class == "LogisticRegression"
    assert model.vocabulary_size == 742
    assert model.prepare_input(X_model).shape == (3, 742)
    assert len(model.predict(X_model)) == 3
    assert model.predict_proba(X_model).shape == (3, 2)
    assert not any(name.startswith("feature_") for name in model.get_feature_names_out())


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
    assert result["feature_semantics"] == "tokens"
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

    def reject_fit(*args, **kwargs):
        raise AssertionError("The fitted S5 text artifact must not be retrained.")

    monkeypatch.setattr(talentsift_model.vectorizer, "fit", reject_fit)
    monkeypatch.setattr(talentsift_model.vectorizer, "fit_transform", reject_fit)
    monkeypatch.setattr(talentsift_model.estimator, "fit", reject_fit)

    result = run_uncertainty(
        talentsift_model,
        X_model,
        y,
        task_type="binary_classification",
        modality="text",
    )

    assert result["status"] == "completed"
    assert result["input_representation"] == "tfidf_sparse"
    assert result["estimator_class"] == "LogisticRegression"
    assert 0.0 <= result["coverage"] <= 1.0


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
