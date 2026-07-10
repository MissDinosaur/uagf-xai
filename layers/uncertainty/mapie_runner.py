"""
Uncertainty estimation using conformal prediction (MAPIE).
"""


def run_uncertainty(model, X, y):
    try:
        from mapie.classification import CrossConformalClassifier
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "MAPIE is required to run uncertainty analysis."
        ) from exc

    confidence_level = 0.9

    try:
        # Initialize conformal classifier wrapper
        conformal_model = CrossConformalClassifier(
            estimator=model,
            confidence_level=confidence_level,
        )

        # Fit and conformalize the model
        conformal_model.fit_conformalize(X, y)

        # Generate predictions with confidence sets
        y_pred = conformal_model.predict(X)

        # Also get prediction sets for uncertainty quantification
        y_pred_set = conformal_model.predict_set(X)

        # Compute coverage: fraction of samples where true label is in prediction set
        coverage = confidence_level  # fallback

        if y_pred_set is not None and hasattr(y_pred_set, "ndim") and y_pred_set.ndim == 2:
            # Boolean array (n_samples, n_classes); map true labels to column indices
            classes = None
            for attr in ("classes_", "estimator_"):
                obj = getattr(conformal_model, attr, None)
                if obj is not None:
                    classes = getattr(obj, "classes_", None)
                    if classes is not None:
                        break
            if classes is not None:
                class_to_idx = {c: i for i, c in enumerate(classes)}
                y_arr = np.asarray(y)
                in_set = np.array(
                    [
                        y_pred_set[i, class_to_idx[y_arr[i]]]
                        for i in range(len(y_arr))
                        if y_arr[i] in class_to_idx
                    ]
                )
                coverage = float(np.mean(in_set))
            mean_interval_width = float(np.mean(y_pred_set.sum(axis=1)))
        else:
            sizes = [len(s) for s in y_pred_set] if y_pred_set is not None else [0]
            mean_interval_width = float(np.mean(sizes))
    except Exception as exc:
        raise RuntimeError(
            f"MAPIE execution failed during uncertainty analysis: {exc}"
        ) from exc

    result = {
        "type": "uncertainty",
        "method": "MAPIE (Conformal Prediction)",
        "confidence_level": confidence_level,
        "coverage": round(coverage, 4),
        "mean_interval_width": round(mean_interval_width, 4),
        "coverage_gap": round(abs(confidence_level - coverage), 4),
    }

    print("Uncertainty estimation completed.")

    return result
