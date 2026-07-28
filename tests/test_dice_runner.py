from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

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

    class FakeData:
        def __init__(self, dataframe, continuous_features, outcome_name):
            self.outcome_name = outcome_name

    class FakeModel:
        def __init__(self, model, backend):
            self.model = model

    class FakeDice:
        def __init__(self, data, model, method):
            self.outcome_name = data.outcome_name

        def generate_counterfactuals(self, query_instance, **options):
            captured_options.update(options)
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

