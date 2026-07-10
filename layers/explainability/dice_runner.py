"""
Counterfactual explanations provide actionable insights by showing
how input features must change to achieve a different prediction
outcome. This enhances user trust and
supports decision transparency.
"""

import json
import os
import pandas as pd
from output_naming import build_output_path


# ---------------------------------------------------------------------------
# Helpers: encode categoricals to int so DiCE (which requires int/float) works,
# and wrap the model to decode them back before calling the underlying estimator.
# ---------------------------------------------------------------------------

def _encode_for_dice(X):
    """Return (X_encoded, cat_dtype_map).

    All pd.Categorical / object columns are replaced by their integer codes.
    cat_dtype_map stores the original CategoricalDtype so we can reconstruct.
    """
    cat_dtype_map = {}
    X_enc = X.copy()
    for col in X_enc.columns:
        if hasattr(X_enc[col], "cat"):               # pd.Categorical dtype
            cat_dtype_map[col] = X_enc[col].dtype
            X_enc[col] = X_enc[col].cat.codes.astype("int64")
        elif X_enc[col].dtype == object:             # plain string column
            as_cat = X_enc[col].astype("category")
            cat_dtype_map[col] = as_cat.dtype
            X_enc[col] = as_cat.cat.codes.astype("int64")
    return X_enc, cat_dtype_map


class _RecodingWrapper:
    """Wraps a model trained on pd.Categorical data so it can accept
    integer-coded DataFrames (as produced by DiCE during CF search)."""

    def __init__(self, model, cat_dtype_map):
        self._model = model
        self._cat_dtype_map = cat_dtype_map

    def _recode(self, X):
        out = pd.DataFrame(X).copy()
        for col, dtype in self._cat_dtype_map.items():
            if col in out.columns:
                out[col] = pd.Categorical.from_codes(
                    out[col].astype(int), dtype.categories
                )
        return out

    def predict(self, X):
        return self._model.predict(self._recode(X))

    def predict_proba(self, X):
        return self._model.predict_proba(self._recode(X))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_dice(model, X, y, provider_name=None, output_namespace="audit"):
    try:
        import dice_ml
    except ImportError as exc:
        raise ImportError(
            "DiCE is required to run counterfactual analysis."
        ) from exc

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "DiCE requires a probabilistic classification model with "
            "predict_proba()."
        )

    os.makedirs("outputs/dice", exist_ok=True)
    output_path = build_output_path(
        "dice",
        provider_name,
        "counterfactuals",
        ".json",
        fallback=output_namespace,
    )

    # Encode pd.Categorical / object columns → int64 so DiCE accepts them
    X_enc, cat_dtype_map = _encode_for_dice(X)

    df = X_enc.copy()
    df["target"] = y

    # Continuous features = everything that is NOT an encoded categorical
    cat_cols = list(cat_dtype_map.keys())
    cont_cols = [c for c in X_enc.columns if c not in cat_cols]

    data = dice_ml.Data(
        dataframe=df,
        continuous_features=cont_cols,
        outcome_name="target",
    )

    # Wrap model only when categorical re-coding is needed (HistGBT case)
    wrapped = _RecodingWrapper(model, cat_dtype_map) if cat_dtype_map else model
    model_dice = dice_ml.Model(model=wrapped, backend="sklearn")

    exp = dice_ml.Dice(data, model_dice, method="random")

    query_instance = X_enc.iloc[0:1]

    dice_exp = exp.generate_counterfactuals(
        query_instance,
        total_CFs=2,
        desired_class="opposite",
    )

    # Extract counterfactuals as a list and save as JSON
    if hasattr(dice_exp, "visualize_as_list"):
        cf_list = dice_exp.visualize_as_list()
    else:
        cf_list = str(dice_exp)

    with open(output_path, "w") as f:
        json.dump({"counterfactuals": str(cf_list)}, f, indent=2)

    result = {
        "type": "counterfactual",
        "method": "DiCE",
        "output": output_path,
        "counterfactuals_count": (
            len(cf_list) if isinstance(cf_list, list) else 1
        ),
    }

    return result
