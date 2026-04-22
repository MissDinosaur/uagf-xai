"""
Counterfactual explanations provide actionable insights by showing
how input features must change to achieve a different prediction
outcome. This enhances user trust and
supports decision transparency.
"""

import dice_ml
import json
import os


def run_dice(model, X, y):

    os.makedirs("outputs/dice", exist_ok=True)

    # Convert to dataframe
    df = X.copy()
    df["target"] = y

    data = dice_ml.Data(
        dataframe=df,
        continuous_features=list(X.columns),
        outcome_name="target"
    )

    model_dice = dice_ml.Model(model=model, backend="sklearn")

    exp = dice_ml.Dice(data, model_dice, method="random")

    query_instance = X.iloc[0:1]

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

    output_path = "outputs/dice/counterfactuals.json"
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