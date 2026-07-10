"""
Validation Domain 1 — German Credit (Statlog) dataset.

Source : OpenML credit-g  (https://www.openml.org/d/31)
Task   : Binary classification — credit risk (good=1 / bad=0)

Sensitive features returned for fairness auditing
  gender    – derived from `personal_status` (values: "female" / "male")
  age_group – binarised age               (values: "young (<=25)" / "adult (>25)")

"""

from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split


def train_model():
    """Load, preprocess, and train on the real German Credit dataset.

    Returns
    -------
    model              : fitted HistGradientBoostingClassifier
    X_test             : pd.DataFrame  (includes gender / age_group columns)
    y_test             : np.ndarray    (0 = bad credit, 1 = good credit)
    sensitive_features : list[str]     (["gender", "age_group"])
    """
    
    # 1. Load dataset
    dataset = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = dataset.frame.copy()

    # 2. Target
    y = (df["class"] == "good").astype(int).values

    # 3. Derive human-readable sensitive features
    #    personal_status encodes sex + marital status in the raw data:
    #      "male div/sep", "female div/dep/mar", "male single",
    #      "male mar/wid"   → all non-female rows are treated as male
    df["gender"] = df["personal_status"].apply(
        lambda v: "female" if str(v).startswith("female") else "male"
    )
    df["age_group"] = df["age"].apply(
        lambda v: "young (<=25)" if int(v) <= 25 else "adult (>25)"
    )
    
    # 4. Build feature matrix
    #    Drop: target, raw columns replaced by derived features above
    X = df.drop(columns=["class"])

    # Cast every string/object column (including gender, age_group) to
    # pd.Categorical so HistGBT picks them up via categorical_features="from_dtype"
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    
    # 5. Train / test split  (stratify to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Train model
    model = HistGradientBoostingClassifier(
        max_iter=100,
        random_state=42,
        categorical_features="from_dtype",
    )
    model.fit(X_train, y_train)

    sensitive_features = ["gender", "age_group"]
    return model, X_test, y_test, sensitive_features
