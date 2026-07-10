from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def train_model():
    X, y = make_regression(n_samples=1000, n_features=10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Regression domain — no demographic sensitive features defined
    sensitive_features = []
    return model, X_test, y_test, sensitive_features