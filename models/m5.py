import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_model():
    df = pd.DataFrame({
        "sales": range(100)
    })

    df["lag1"] = df["sales"].shift(1)
    df["lag2"] = df["sales"].shift(2)
    df = df.dropna()

    X = df[["lag1", "lag2"]]
    y = df["sales"]

    split = int(len(df) * 0.8)

    model = RandomForestRegressor()
    model.fit(X[:split], y[:split])

    return model, X[split:], y[split:]