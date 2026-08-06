"""Traditional ML model loading and sklearn bundle normalization."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .artifact_utils import unwrap_artifact

try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:  # pragma: no cover - older sklearn or minimal envs
    InconsistentVersionWarning = None


class EncodedSklearnModel(BaseEstimator, ClassifierMixin):
    """Wrap an sklearn estimator saved with separate categorical encoders."""

    def __init__(self, model, encoders=None, feature_cols=None, artifact_label=None):
        self.model = model
        self.encoders = encoders
        self.feature_cols = feature_cols
        self.artifact_label = artifact_label
        self._encoders = encoders or {}
        self._feature_cols = (
            list(feature_cols)
            if feature_cols is not None
            else list(getattr(model, "feature_names_in_", []))
        )
        self._artifact_label = artifact_label or "unknown_artifact"

    @property
    def estimator(self):
        """Return the underlying sklearn estimator."""
        return self.model

    def prepare_input(self, X):
        """Convert raw feature data into model-ready numeric data."""
        if not isinstance(X, pd.DataFrame):
            if self._feature_cols:
                X = pd.DataFrame(X, columns=self._feature_cols)
            else:
                X = pd.DataFrame(X)

        X_prepared = X.copy()
        if self._feature_cols:
            missing_cols = [
                column
                for column in self._feature_cols
                if column not in X_prepared.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"[{self._artifact_label}] Input data is missing required "
                    f"model features: {missing_cols}"
                )
            X_prepared = X_prepared[self._feature_cols]

        for column, encoder in self._encoders.items():
            if column not in X_prepared.columns:
                continue

            series = X_prepared[column]
            known_classes = list(map(str, getattr(encoder, "classes_", [])))
            known_class_set = set(known_classes)
            missing_mask = series.isna()
            if missing_mask.any():
                placeholder = self._select_missing_placeholder(known_classes)
                if placeholder is None:
                    raise ValueError(
                        f"[{self._artifact_label}] Column '{column}' contains "
                        f"{int(missing_mask.sum())} missing value(s), but the "
                        "saved encoder does not define a safe placeholder "
                        "category for them."
                    )
                series = series.copy()
                series.loc[missing_mask] = placeholder

            values = series.astype(str)
            observed_classes = set(values.dropna().unique())
            unseen_classes = observed_classes - known_class_set
            if unseen_classes:
                raise ValueError(
                    f"[{self._artifact_label}] Column '{column}' contains unseen "
                    "categories not known to the saved encoder: "
                    f"{sorted(unseen_classes)}"
                )
            X_prepared[column] = encoder.transform(values)

        return X_prepared

    def _prepare_X(self, X):
        """Backward-compatible alias for prepare_input()."""
        return self.prepare_input(X)

    @staticmethod
    def _select_missing_placeholder(known_classes: list[str]) -> str | None:
        preferred_placeholders = [
            "None",
            "none",
            "missing",
            "Missing",
            "unknown",
            "Unknown",
            "NA",
            "N/A",
            "nan",
        ]
        known_class_set = set(known_classes)
        for candidate in preferred_placeholders:
            if candidate in known_class_set:
                return candidate
        return None

    def fit(self, X, y=None, **fit_params):
        X_prepared = self.prepare_input(X)
        if not hasattr(self.model, "fit"):
            raise AttributeError(
                f"[{self._artifact_label}] The underlying model does not "
                "support fit()."
            )
        fitted_model = self.model.fit(X_prepared, y, **fit_params)
        if fitted_model is not None:
            self.model = fitted_model
        return self

    def predict(self, X):
        return self.model.predict(self.prepare_input(X))

    def predict_proba(self, X):
        X_prepared = self.prepare_input(X)
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                f"[{self._artifact_label}] The underlying model does not "
                "support predict_proba()."
            )
        return self.model.predict_proba(X_prepared)

    def decision_function(self, X):
        X_prepared = self.prepare_input(X)
        if not hasattr(self.model, "decision_function"):
            raise AttributeError(
                f"[{self._artifact_label}] The underlying model does not "
                "support decision_function()."
            )
        return self.model.decision_function(X_prepared)

    def __call__(self, X):
        return self.predict(X)

    def __getattr__(self, item):
        return getattr(self.model, item)

    def __sklearn_clone__(self):
        """Preserve fitted encoders when external libraries clone the wrapper."""
        return type(self)(
            model=clone(self.model),
            encoders=self.encoders,
            feature_cols=self.feature_cols,
            artifact_label=self.artifact_label,
        )


class SklearnTextModelAdapter:
    """Adapt raw text shapes while delegating inference to the S5 Pipeline.

    The adapter never copies fitted parameters or reproduces classifier logic.
    Predictions use the original fitted Pipeline, while model-ready TF-IDF input
    for explainers uses the original fitted vectorizer's transform() method.
    """

    def __init__(self, pipeline, vectorizer, estimator, text_feature_column):
        self.underlying_pipeline = pipeline
        self.vectorizer = vectorizer
        self.estimator = estimator
        self.text_feature_column = text_feature_column
        self.input_adapter = "sklearn_text"
        self.vectorizer_class = type(vectorizer).__name__
        self.estimator_class = type(estimator).__name__
        self.vocabulary_size = len(vectorizer.vocabulary_)

    @classmethod
    def from_artifact(cls, artifact):
        """Return an adapter only for the supported fitted sklearn text shape."""
        if not isinstance(artifact, Pipeline) or not artifact.steps:
            return None

        estimator = artifact.steps[-1][1]
        if not isinstance(estimator, LogisticRegression):
            return None
        if not hasattr(estimator, "coef_") or not hasattr(estimator, "classes_"):
            return None

        candidates = []
        for _, component in artifact.steps[:-1]:
            if not isinstance(component, ColumnTransformer):
                continue
            for _, transformer, columns in component.transformers_:
                if isinstance(transformer, TfidfVectorizer):
                    candidates.append((transformer, columns))
                elif isinstance(transformer, Pipeline):
                    for _, nested in transformer.steps:
                        if isinstance(nested, TfidfVectorizer):
                            candidates.append((nested, columns))

        if len(candidates) != 1:
            return None
        vectorizer, columns = candidates[0]
        if not hasattr(vectorizer, "vocabulary_"):
            return None
        if isinstance(columns, str):
            text_columns = [columns]
        elif isinstance(columns, (list, tuple, np.ndarray)):
            text_columns = list(columns)
        else:
            return None
        if len(text_columns) != 1 or not isinstance(text_columns[0], str):
            return None
        return cls(artifact, vectorizer, estimator, text_columns[0])

    @property
    def classes_(self):
        return self.estimator.classes_

    def _raw_texts(self, X) -> list[str]:
        if isinstance(X, pd.DataFrame):
            if self.text_feature_column not in X.columns:
                raise ValueError(
                    "Text model input is missing required feature column "
                    f"{self.text_feature_column!r}."
                )
            values = X[self.text_feature_column].tolist()
        elif isinstance(X, pd.Series):
            values = X.tolist()
        elif isinstance(X, (list, tuple, np.ndarray)):
            values = np.asarray(X, dtype=object).reshape(-1).tolist()
        else:
            raise TypeError(
                "Text model input must be a one-column DataFrame, Series, "
                "or sequence of strings."
            )
        if any(not isinstance(value, str) for value in values):
            raise ValueError("Text model input must contain only non-null strings.")
        return values

    def _pipeline_frame(self, X) -> pd.DataFrame:
        return pd.DataFrame({self.text_feature_column: self._raw_texts(X)})

    def prepare_input(self, X):
        """Transform raw text with the fitted TF-IDF vectorizer."""
        return self.vectorizer.transform(self._raw_texts(X))

    def predict(self, X):
        return self.underlying_pipeline.predict(self._pipeline_frame(X))

    def predict_proba(self, X):
        return self.underlying_pipeline.predict_proba(self._pipeline_frame(X))

    def decision_function(self, X):
        return self.underlying_pipeline.decision_function(self._pipeline_frame(X))

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()


class TraditionalModelLoader:
    """Load serialized Traditional ML models and normalize model bundles."""

    SERIALIZED_FORMATS = frozenset({"joblib", "pkl", "pickle"})
    SERIALIZED_FRAMEWORKS = frozenset(
        {"sklearn", "scikit-learn", "xgboost", "lightgbm", "catboost"}
    )
    SERIALIZED_SUFFIXES = frozenset({".joblib", ".pkl", ".pickle"})

    @classmethod
    def supports(cls, source_path: Path, model_format: str, model_framework: str) -> bool:
        """Return whether the resolved artifact is a supported serialized model."""
        if not source_path.is_file():
            return False
        return (
            source_path.suffix.lower() in cls.SERIALIZED_SUFFIXES
            or model_format in cls.SERIALIZED_FORMATS
            or model_framework in cls.SERIALIZED_FRAMEWORKS
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        uri: str | None = None,
        model_format: str | None = None,
        model_framework: str | None = None,
        model_type: str | None = None,
        model_entrypoint: str | None = None,
    ):
        """Load a serialized model file and preserve artifact diagnostics."""
        if not path.exists():
            raise FileNotFoundError(
                "Serialized model artifact not found. "
                f"uri={uri!r}, resolved_path={str(path)!r}, "
                f"model_format={model_format!r}, "
                f"model_framework={model_framework!r}, "
                f"model_type={model_type!r}, "
                f"model_entrypoint={model_entrypoint!r}"
            )

        joblib_error = None
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                loaded = joblib.load(path)
            except Exception as exc:
                joblib_error = exc
                try:
                    with open(path, "rb") as model_file:
                        loaded = pickle.load(model_file)
                except Exception as pickle_error:
                    raise RuntimeError(
                        "Failed to load serialized model artifact. "
                        f"uri={uri!r}, resolved_path={str(path)!r}, "
                        f"model_format={model_format!r}, "
                        f"model_framework={model_framework!r}, "
                        f"model_type={model_type!r}, "
                        f"model_entrypoint={model_entrypoint!r}, "
                        f"joblib_error={type(joblib_error).__name__}: {joblib_error}, "
                        f"pickle_error={type(pickle_error).__name__}: {pickle_error}"
                    ) from pickle_error

        load_warnings = cls._collect_version_warnings(
            caught_warnings,
            uri=uri,
            path=path,
        )
        artifact_model = loaded.get("model") if isinstance(loaded, dict) else loaded
        text_adapter = SklearnTextModelAdapter.from_artifact(artifact_model)
        wrapped_bundle = cls._wrap_bundle(
            loaded,
            uri=uri,
            path=path,
            model_format=model_format,
            model_framework=model_framework,
            model_type=model_type,
            model_entrypoint=model_entrypoint,
        )
        if text_adapter is not None:
            final_model = text_adapter
            print(
                "[ModelLoader] Loaded fitted sklearn text model adapter "
                f"with vectorizer={text_adapter.vectorizer_class}, "
                f"estimator={text_adapter.estimator_class}, "
                f"text_feature={text_adapter.text_feature_column!r}, "
                f"vocabulary_size={text_adapter.vocabulary_size}."
            )
        else:
            final_model = wrapped_bundle or unwrap_artifact(loaded)
        if load_warnings:
            try:
                setattr(final_model, "_uagf_load_warnings", load_warnings)
            except Exception:
                # Report metadata is best-effort and must not alter model loading.
                pass
        return final_model

    @staticmethod
    def _collect_version_warnings(caught_warnings, *, uri, path) -> list[str]:
        load_warnings: list[str] = []
        for warning_item in caught_warnings:
            if (
                InconsistentVersionWarning is not None
                and warning_item.category is InconsistentVersionWarning
            ):
                load_warnings.append(
                    "The model artifact was serialized with a different "
                    "scikit-learn version; compatibility should be reviewed."
                )
                print(
                    "[ModelLoader] Artifact was serialized with a different "
                    "scikit-learn version; continuing with caution. "
                    f"uri={uri!r}, resolved_path={str(path)!r}"
                )
                break
        return load_warnings

    @staticmethod
    def _wrap_bundle(
        artifact,
        *,
        uri: str | None = None,
        path: Path | None = None,
        model_format: str | None = None,
        model_framework: str | None = None,
        model_type: str | None = None,
        model_entrypoint: str | None = None,
    ):
        if not isinstance(artifact, dict) or "model" not in artifact:
            return None

        model = artifact["model"]
        encoders = artifact.get("encoders", {}) or {}
        feature_cols = artifact.get("feature_cols")
        algorithm = artifact.get("algorithm")
        version = artifact.get("version")
        artifact_label = (
            f"uri={uri!r}, resolved_path={str(path)!r}, "
            f"model_format={model_format!r}, model_framework={model_framework!r}, "
            f"model_type={model_type!r}, model_entrypoint={model_entrypoint!r}, "
            f"algorithm={algorithm!r}, version={version!r}"
        )
        print(
            "[ModelLoader] Loaded sklearn model bundle "
            f"with model={type(model).__name__}, "
            f"encoders={list(encoders.keys())}, "
            f"feature_cols={len(feature_cols or [])}, "
            f"uri={uri!r}, resolved_path={str(path)!r}"
        )
        return EncodedSklearnModel(
            model=model,
            encoders=encoders,
            feature_cols=feature_cols,
            artifact_label=artifact_label,
        )
