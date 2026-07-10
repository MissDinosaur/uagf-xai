"""
Model loading utilities.

The loader uses two stable metadata fields from S5:

- model_format
- model_framework

This keeps the loading contract explicit and avoids the older
loader hint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings
import joblib
import pickle

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from .artifact_utils import resolve_artifact_path, unwrap_artifact

try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:  # pragma: no cover - older sklearn or minimal envs
    InconsistentVersionWarning = None


class EncodedSklearnModel(BaseEstimator, ClassifierMixin):
    """Wrapper for sklearn models saved with separate categorical encoders."""

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
                col for col in self._feature_cols
                if col not in X_prepared.columns
            ]

            if missing_cols:
                raise ValueError(
                    f"[{self._artifact_label}] Input data is missing required "
                    f"model features: {missing_cols}"
                )

            X_prepared = X_prepared[self._feature_cols]

        for col, encoder in self._encoders.items():
            if col not in X_prepared.columns:
                continue

            values = X_prepared[col].astype(str)
            known_classes = set(map(str, getattr(encoder, "classes_", [])))
            observed_classes = set(values.dropna().unique())
            unseen_classes = observed_classes - known_classes

            if unseen_classes:
                raise ValueError(
                    f"[{self._artifact_label}] Column '{col}' contains unseen "
                    f"categories not known to the saved encoder: "
                    f"{sorted(unseen_classes)}"
                )

            X_prepared[col] = encoder.transform(values)

        return X_prepared

    def _prepare_X(self, X):
        """Backward-compatible alias for prepare_input()."""
        return self.prepare_input(X)

    def fit(self, X, y=None, **fit_params):
        X_prepared = self.prepare_input(X)
        if not hasattr(self.model, "fit"):
            raise AttributeError(
                f"[{self._artifact_label}] The underlying model does not "
                f"support fit()."
            )

        fitted_model = self.model.fit(X_prepared, y, **fit_params)
        if fitted_model is not None:
            self.model = fitted_model
        return self

    def predict(self, X):
        X_prepared = self.prepare_input(X)
        return self.model.predict(X_prepared)

    def predict_proba(self, X):
        X_prepared = self.prepare_input(X)

        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                f"[{self._artifact_label}] The underlying model does not "
                f"support predict_proba()."
            )

        return self.model.predict_proba(X_prepared)

    def decision_function(self, X):
        X_prepared = self.prepare_input(X)

        if not hasattr(self.model, "decision_function"):
            raise AttributeError(
                f"[{self._artifact_label}] The underlying model does not "
                f"support decision_function()."
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


class ModelLoader:
    """
    Loads a trained model from the URI and metadata provided by S5.
    """

    @staticmethod
    def load(audit_context):
        uri = getattr(audit_context, "model_artifact_uri", None)
        if not uri:
            raise ValueError(
                "audit_context.model_artifact_uri is required to load a model."
            )

        path = resolve_artifact_path(uri)
        model_format, model_framework = ModelLoader._resolve_model_metadata(
            audit_context
        )

        if ModelLoader._is_huggingface_stack(model_format, model_framework):
            return ModelLoader._load_huggingface_model(
                path,
                uri=uri,
                model_format=model_format,
                model_framework=model_framework,
            )

        if ModelLoader._is_serialized_stack(model_format, model_framework):
            return ModelLoader._load_serialized_model(
                path,
                uri=uri,
                model_format=model_format,
                model_framework=model_framework,
            )

        if model_format == "onnx" or model_framework == "onnxruntime":
            raise NotImplementedError(
                "ONNX loader is not implemented yet. "
                f"uri={uri!r}, resolved_path={str(path)!r}"
            )

        raise ValueError(
            "Unsupported model metadata for loading a model artifact. "
            f"uri={uri!r}, resolved_path={str(path)!r}, "
            f"model_format={model_format!r}, "
            f"model_framework={model_framework!r}"
        )

    @staticmethod
    def load_tokenizer(audit_context):
        """
        Load a HuggingFace tokenizer when the artifact metadata indicates an
        LLM / transformer stack. Return None for non-LLM models.
        """
        model_format, model_framework = ModelLoader._resolve_model_metadata(
            audit_context
        )
        if not ModelLoader._is_huggingface_stack(model_format, model_framework):
            return None

        uri = getattr(audit_context, "model_artifact_uri", None)
        if not uri:
            raise ValueError(
                "audit_context.model_artifact_uri is required to load a tokenizer."
            )

        path = resolve_artifact_path(uri)
        return ModelLoader._load_huggingface_tokenizer(
            path,
            uri=uri,
            model_format=model_format,
            model_framework=model_framework,
        )

    @staticmethod
    def _normalize(value: Any) -> str:
        return str(value).strip().lower() if value is not None else ""

    @staticmethod
    def _resolve_model_metadata(audit_context) -> tuple[str, str]:
        model_format = ModelLoader._normalize(
            getattr(audit_context, "model_format", None)
        )
        model_framework = ModelLoader._normalize(
            getattr(audit_context, "model_framework", None)
        )
        system_type = ModelLoader._normalize(
            getattr(audit_context, "system_type", "")
        )

        if not model_format and not model_framework:
            if system_type in {"llm", "agentic"}:
                return "huggingface", "transformers"
            return "joblib", "sklearn"

        if not model_format:
            model_format = ModelLoader._infer_format_from_framework(
                model_framework
            )

        if not model_framework:
            model_framework = ModelLoader._infer_framework_from_format(
                model_format
            )

        return model_format, model_framework

    @staticmethod
    def _infer_format_from_framework(model_framework: str) -> str:
        if model_framework in {"sklearn", "scikit-learn", "xgboost", "lightgbm"}:
            return "joblib"
        if model_framework in {"transformers", "huggingface"}:
            return "huggingface"
        if model_framework in {"onnxruntime"}:
            return "onnx"
        return model_framework

    @staticmethod
    def _infer_framework_from_format(model_format: str) -> str:
        if model_format in {"joblib", "pkl", "pickle"}:
            return "sklearn"
        if model_format in {"huggingface", "hf"}:
            return "transformers"
        if model_format == "onnx":
            return "onnxruntime"
        return model_format

    @staticmethod
    def _is_serialized_stack(model_format: str, model_framework: str) -> bool:
        serialized_formats = {"joblib", "pkl", "pickle"}
        serialized_frameworks = {
            "sklearn",
            "scikit-learn",
            "xgboost",
            "lightgbm",
            "catboost",
        }
        return (
            model_format in serialized_formats
            or model_framework in serialized_frameworks
        )

    @staticmethod
    def _is_huggingface_stack(model_format: str, model_framework: str) -> bool:
        return model_format in {"huggingface", "hf"} or model_framework in {
            "huggingface",
            "transformers",
        }

    @staticmethod
    def _load_serialized_model(
        path: Path,
        uri: str | None = None,
        model_format: str | None = None,
        model_framework: str | None = None,
    ):
        if not path.exists():
            raise FileNotFoundError(
                "Serialized model artifact not found. "
                f"uri={uri!r}, resolved_path={str(path)!r}, "
                f"model_format={model_format!r}, "
                f"model_framework={model_framework!r}"
            )

        joblib_error = None
        pickle_error = None
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                loaded = joblib.load(path)
            except Exception as exc:
                joblib_error = exc
                try:
                    with open(path, "rb") as f:
                        loaded = pickle.load(f)
                except Exception as pickle_exc:
                    pickle_error = pickle_exc
                    raise RuntimeError(
                        "Failed to load serialized model artifact. "
                        f"uri={uri!r}, resolved_path={str(path)!r}, "
                        f"model_format={model_format!r}, "
                        f"model_framework={model_framework!r}, "
                        f"joblib_error={type(joblib_error).__name__}: {joblib_error}, "
                        f"pickle_error={type(pickle_error).__name__}: {pickle_error}"
                    ) from pickle_exc

        for warning_item in caught_warnings:
            if (
                InconsistentVersionWarning is not None
                and warning_item.category is InconsistentVersionWarning
            ):
                print(
                    "[ModelLoader] Artifact was serialized with a different "
                    "scikit-learn version; continuing with caution. "
                    f"uri={uri!r}, resolved_path={str(path)!r}"
                )
                break

        wrapped_bundle = ModelLoader._wrap_encoded_sklearn_bundle(
            loaded,
            uri=uri,
            path=path,
            model_format=model_format,
            model_framework=model_framework,
        )
        if wrapped_bundle is not None:
            return wrapped_bundle

        return unwrap_artifact(loaded)

    @staticmethod
    def _wrap_encoded_sklearn_bundle(
        artifact,
        uri: str | None = None,
        path: Path | None = None,
        model_format: str | None = None,
        model_framework: str | None = None,
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

    @staticmethod
    def _load_huggingface_model(
        path: Path,
        uri: str | None = None,
        model_format: str | None = None,
        model_framework: str | None = None,
    ):
        try:
            from transformers import AutoModelForCausalLM
            from transformers import AutoTokenizer
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required to load HuggingFace models. "
                f"uri={uri!r}, resolved_path={str(path)!r}, "
                f"model_format={model_format!r}, "
                f"model_framework={model_framework!r}"
            ) from exc

        source = path if path.is_dir() else path.parent
        if not source.exists():
            raise FileNotFoundError(
                "HuggingFace model source directory not found. "
                f"uri={uri!r}, resolved_path={str(path)!r}, "
                f"source_dir={str(source)!r}, "
                f"model_format={model_format!r}, "
                f"model_framework={model_framework!r}"
            )

        model = AutoModelForCausalLM.from_pretrained(str(source))
        tokenizer = AutoTokenizer.from_pretrained(str(source))
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    @staticmethod
    def _load_huggingface_tokenizer(
        path: Path,
        uri: str | None = None,
        model_format: str | None = None,
        model_framework: str | None = None,
    ):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required to load HuggingFace tokenizers. "
                f"uri={uri!r}, resolved_path={str(path)!r}, "
                f"model_format={model_format!r}, "
                f"model_framework={model_framework!r}"
            ) from exc

        source = path if path.is_dir() else path.parent
        if not source.exists():
            raise FileNotFoundError(
                "HuggingFace tokenizer source directory not found. "
                f"uri={uri!r}, resolved_path={str(path)!r}, "
                f"source_dir={str(source)!r}, "
                f"model_format={model_format!r}, "
                f"model_framework={model_framework!r}"
            )

        return AutoTokenizer.from_pretrained(str(source))
