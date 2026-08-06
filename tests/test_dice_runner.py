from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest

from adapters.s5_audit_adapter import AuditAdapter
from api.audit_api import _load_s5_resources
from layers.explainability import dice_runner


class _ThresholdModel:
    def predict(self, X):
        return (pd.DataFrame(X)["income"].astype(float).to_numpy() >= 50_000).astype(int)

    def predict_proba(self, X):
        prediction = self.predict(X).astype(float)
        return np.column_stack([1.0 - prediction, prediction])


def test_dice_writes_structured_counterfactual_artifact(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "counterfactuals.json"
    captured_options = {}
    captured_schema = {}

    class FakeData:
        def __init__(
            self, dataframe, continuous_features, outcome_name, permitted_range
        ):
            self.outcome_name = outcome_name
            captured_schema["reference_dtypes"] = {
                column: str(dataframe[column].dtype)
                for column in dataframe.columns
                if column != outcome_name
            }
            captured_schema["continuous_features"] = continuous_features
            captured_schema["permitted_range"] = permitted_range

    class FakeModel:
        def __init__(self, model, backend):
            self.model = model

    class FakeDice:
        def __init__(self, data, model, method):
            self.outcome_name = data.outcome_name

        def generate_counterfactuals(self, query_instance, **options):
            captured_options.update(options)
            captured_schema["query_dtypes"] = {
                column: str(dtype) for column, dtype in query_instance.dtypes.items()
            }
            counterfactual = query_instance.copy()
            counterfactual.loc[:, "income"] = 40_000
            counterfactual[self.outcome_name] = 0
            return SimpleNamespace(
                cf_examples_list=[
                    SimpleNamespace(final_cfs_df=counterfactual)
                ]
            )

    fake_dice_ml = SimpleNamespace(Data=FakeData, Model=FakeModel, Dice=FakeDice)
    monkeypatch.setitem(sys.modules, "dice_ml", fake_dice_ml)
    monkeypatch.setattr(
        dice_runner,
        "build_output_path",
        lambda *args, **kwargs: str(output_path),
    )

    X = pd.DataFrame(
        {
            "income": [80_000, 30_000],
            "personal_status": ["single", "married"],
            "foreign_worker": ["yes", "no"],
        }
    )
    result = dice_runner.run_dice(
        _ThresholdModel(),
        X,
        pd.Series([1, 0]),
        sensitive_features=["personal_status", "foreign_worker"],
    )
    artifact = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["type"] == "explainability"
    assert result["evidence_type"] == "counterfactual_explanation"
    assert artifact["query_instance"]["personal_status"] == "single"
    assert artifact["original_prediction"] == 1
    assert artifact["desired_class"] == "opposite"
    assert artifact["counterfactual_prediction"] == 0
    assert isinstance(artifact["counterfactuals"], list)
    assert isinstance(artifact["counterfactuals"][0], dict)
    assert artifact["counterfactuals"][0]["changed_features"] == [
        {
            "feature": "income",
            "feature_type": "numeric",
            "original_value": 80_000,
            "counterfactual_value": 40_000,
            "delta": -40_000.0,
            "model_ready_original_value": 80_000,
            "model_ready_counterfactual_value": 40_000,
            "is_sensitive_or_immutable": False,
        }
    ]
    assert "personal_status" not in captured_options["features_to_vary"]
    assert "foreign_worker" not in captured_options["features_to_vary"]
    assert captured_schema["reference_dtypes"] == captured_schema["query_dtypes"]
    assert captured_schema["reference_dtypes"]["income"] == "float64"
    assert captured_schema["reference_dtypes"]["personal_status"] == "object"
    assert captured_schema["continuous_features"] == ["income"]
    assert all(
        isinstance(value, str)
        for values in captured_schema["permitted_range"].values()
        for value in values
    )


def test_counterfactual_feature_policy_is_explicit_and_validated():
    policy = dice_runner._resolve_feature_policy(
        ["income", "age", "credit_history", "personal_status"],
        ["personal_status"],
        ["income"],
        ["age", "credit_history"],
        "credit_risk",
    )

    assert policy["features_to_vary"] == ["income"]
    assert policy["excluded_sensitive_features"] == ["personal_status"]
    assert policy["excluded_immutable_features"] == ["age", "credit_history"]
    assert policy["counterfactual_policy_source"] == "s5_actionable_allowlist"
    assert policy["counterfactual_policy_status"] == "validated"


@pytest.mark.parametrize(
    "actionable,immutable,expected",
    [
        (["unknown"], [], "unknown model features"),
        (["income", "income"], [], "duplicate columns"),
        (["credit_risk"], [], "target_column"),
        (["personal_status"], [], "must not authorize S5 sensitive features"),
    ],
)
def test_counterfactual_feature_policy_rejects_invalid_contracts(
    actionable, immutable, expected
):
    with pytest.raises(ValueError, match=expected):
        dice_runner._resolve_feature_policy(
            ["income", "personal_status"],
            ["personal_status"],
            actionable,
            immutable,
            "credit_risk",
        )


def test_no_authorized_features_returns_structured_skip(tmp_path, monkeypatch):
    output_path = tmp_path / "counterfactuals.json"
    monkeypatch.setitem(sys.modules, "dice_ml", SimpleNamespace())
    monkeypatch.setattr(
        dice_runner,
        "build_output_path",
        lambda *args, **kwargs: str(output_path),
    )

    result = dice_runner.run_dice(
        _ThresholdModel(),
        pd.DataFrame({"income": [80_000, 30_000]}),
        pd.Series([1, 0]),
        actionable_features=[],
    )

    assert result["status"] == "skipped"
    assert result["features_to_vary"] == []
    assert result["policy_violation_detected"] is False
    assert "authorizes no model features" in result["reason"]
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "skipped"


def test_finclear_dice_uses_saved_model_and_case_policy(tmp_path, monkeypatch):
    with open(
        "data/01_finclear_gmbh/s5_finclear_gmbh_audit_state.json",
        encoding="utf-8",
    ) as input_file:
        context = AuditAdapter.from_s5_state(json.load(input_file))
    bundle, views = _load_s5_resources(context)
    output_path = tmp_path / "finclear_counterfactuals.json"
    monkeypatch.setattr(
        dice_runner,
        "build_output_path",
        lambda *args, **kwargs: str(output_path),
    )

    fitted_objects = [bundle.model.estimator, *bundle.model.encoders.values()]
    before_hash = joblib.hash(fitted_objects)
    classes = {type(item) for item in fitted_objects}
    with ExitStack() as stack:
        for cls in classes:
            for method_name in ("fit", "fit_transform", "partial_fit"):
                if hasattr(cls, method_name):
                    stack.enter_context(
                        patch.object(
                            cls,
                            method_name,
                            side_effect=AssertionError(
                                f"{method_name} must not be called"
                            ),
                        )
                    )
        predict_spy = stack.enter_context(
            patch.object(
                bundle.model.estimator,
                "predict",
                wraps=bundle.model.estimator.predict,
            )
        )
        predict_proba_spy = stack.enter_context(
            patch.object(
                bundle.model.estimator,
                "predict_proba",
                wraps=bundle.model.estimator.predict_proba,
            )
        )
        result = dice_runner.run_dice(
            bundle.model,
            views.X_model,
            views.y,
            sensitive_features=views.sensitive_feature_columns,
            actionable_features=context.counterfactual_actionable_feature_columns,
            immutable_features=context.counterfactual_immutable_feature_columns,
            target_column=context.target_column,
        )
    after_hash = joblib.hash(fitted_objects)

    excluded = {"personal_status", "foreign_worker", "age", "credit_history"}
    expected_features_to_vary = [
        "checking_status",
        "duration_months",
        "purpose",
        "credit_amount",
        "savings_account",
        "employment_since",
        "installment_rate_pct",
        "other_debtors",
        "residence_since",
        "property",
        "other_installments",
        "housing",
        "existing_credits",
        "job",
        "num_dependents",
        "telephone",
    ]
    changed = {
        change["feature"]
        for item in result["counterfactuals"]
        for change in item["changed_features"]
    }
    assert result.get("status", "completed") == "completed"
    assert result["excluded_sensitive_features"] == ["personal_status", "foreign_worker"]
    assert result["excluded_immutable_features"] == ["credit_history", "age"]
    assert result["features_to_vary"] == expected_features_to_vary
    assert changed.isdisjoint(excluded)
    assert changed.issubset(set(result["features_to_vary"]))
    assert result["policy_violation_detected"] is False
    assert result["sensitive_or_immutable_change_detected"] is False
    assert all(
        item["counterfactual_prediction"] != result["original_prediction"]
        for item in result["counterfactuals"]
    )
    assert predict_spy.called
    assert predict_proba_spy.called
    assert before_hash == after_hash
