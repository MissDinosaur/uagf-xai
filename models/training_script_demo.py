import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data():
    df = pd.read_csv("data/german_credit.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train():
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/german_credit.pkl")

    return model, X_test, y_test


if __name__ == "__main__":
    model, X_test, y_test = train()

    from uagf.audit import audit

    audit(
        model=model,
        data=X_test,
        target=y_test,
        risk_level="high"
    )