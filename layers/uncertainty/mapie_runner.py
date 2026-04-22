"""
Uncertainty estimation using conformal prediction (MAPIE).
"""

from mapie.classification import CrossConformalClassifier
import numpy as np


def run_uncertainty(model, X, y):

    # Initialize conformal classifier wrapper
    conformal_model = CrossConformalClassifier(estimator=model, confidence_level=0.9)

    # Fit and conformalize the model
    conformal_model.fit_conformalize(X, y)

    # Generate predictions with confidence sets
    y_pred = conformal_model.predict(X)

    # Also get prediction sets for uncertainty quantification
    y_pred_set = conformal_model.predict_set(X)

    # Calculate average set size as a measure of uncertainty
    # Larger sets = higher uncertainty
    avg_set_size = np.mean([len(s) for s in y_pred_set]) if y_pred_set is not None else 0

    result = {
        "type": "uncertainty",
        "method": "MAPIE (Conformal Prediction)",
        "predictions": y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
        "average_prediction_set_size": float(avg_set_size),
        "confidence_level": 0.9
    }

    print("Uncertainty estimation completed.")

    return result
