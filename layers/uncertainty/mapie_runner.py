"""
Uncertainty estimation using conformal prediction (MAPIE).
"""

from mapie.classification import MapieClassifier


def run_uncertainty(model, X, y):

    mapie = MapieClassifier(estimator=model)

    mapie.fit(X, y)

    predictions, confidence = mapie.predict(X, alpha=0.1)

    print("Uncertainty estimation completed.")
