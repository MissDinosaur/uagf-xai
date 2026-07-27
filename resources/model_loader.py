"""
Model loading utilities.

The loader uses stable metadata fields from S5 and supports two artifact
shapes:

1. Single-file serialized models such as .joblib / .pkl
2. Model directories that require an explicit model_entrypoint
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import warnings
import joblib
import pickle

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from dataclasses import dataclass, field

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

            series = X_prepared[col]
            known_classes = list(map(str, getattr(encoder, "classes_", [])))
            known_class_set = set(known_classes)
            missing_mask = series.isna()

            if missing_mask.any():
                placeholder = self._select_missing_placeholder(
                    known_classes
                )
                if placeholder is None:
                    raise ValueError(
                        f"[{self._artifact_label}] Column '{col}' contains "
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
                    f"[{self._artifact_label}] Column '{col}' contains unseen "
                    f"categories not known to the saved encoder: "
                    f"{sorted(unseen_classes)}"
                )

            X_prepared[col] = encoder.transform(values)

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


@dataclass
class MetadataOnlyLLMArtifact:
    """
    Structured placeholder for metadata-only LLM artifacts.

    The object is intentionally non-callable so evidence runners can detect
    that execution-level inference is not possible and skip gracefully.
    """

    status: str = "metadata_only"
    is_loadable: bool = False
    resolved_path: str | None = None
    model_type: str | None = None
    model_framework: str | None = None
    model_entrypoint: str | None = None
    loaded_metadata_files: list[str] = field(default_factory=list)
    reason: str = (
        "The provided LLM artifact contains configuration metadata but no "
        "loadable model weights."
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __call__(self, *args, **kwargs):  # pragma: no cover - defensive guard
        raise RuntimeError(
            "This LLM artifact is metadata-only and cannot be executed. "
            f"resolved_path={self.resolved_path!r}, "
            f"model_framework={self.model_framework!r}, "
            f"model_type={self.model_type!r}"
        )


class ModelLoader:
    """
    Loads a trained model from the URI and metadata provided by S5.
    """

    @staticmethod
    def load(audit_context):
        resolved = ModelLoader._resolve_artifact(audit_context)
        artifact_path = resolved["artifact_path"]
        source_path = resolved["source_path"]
        model_format = resolved["model_format"]
        model_framework = resolved["model_framework"]
        model_type = resolved["model_type"]
        model_entrypoint = resolved["model_entrypoint"]

        if not resolved["is_directory_contract"] and not source_path.exists():
            raise FileNotFoundError(
                "Model artifact not found. "
                f"uri={resolved['uri']!r}, resolved_path={str(source_path)!r}, "
                f"model_format={model_format!r}, "
                f"model_framework={model_framework!r}, "
                f"model_type={model_type!r}, "
                f"model_entrypoint={model_entrypoint!r}"
            )

        if resolved["is_directory_contract"]:
            print(
                "[ModelLoader] Loaded model directory: "
                f"uri={resolved['uri']!r}, resolved_path={str(artifact_path)!r}"
            )
            print(
                "[ModelLoader] Model entrypoint: "
                f"{model_entrypoint!r} -> {str(source_path)!r}"
            )
        else:
            print(
                "[ModelLoader] Loaded single-file model artifact: "
                f"uri={resolved['uri']!r}, resolved_path={str(source_path)!r}"
            )

        if ModelLoader._is_huggingface_stack(model_format, model_framework):
            if source_path.is_dir():
                return ModelLoader._load_huggingface_model(
                    source_path,
                    uri=resolved["uri"],
                    model_format=model_format,
                    model_framework=model_framework,
                    model_type=model_type,
                    model_entrypoint=model_entrypoint,
                )

            if resolved["is_directory_contract"]:
                raise FileNotFoundError(
                    "HuggingFace model directory could not be resolved. "
                    f"uri={resolved['uri']!r}, resolved_path={str(artifact_path)!r}, "
                    f"entrypoint={model_entrypoint!r}, "
                    f"source_path={str(source_path)!r}"
                )

        if ModelLoader._should_load_serialized_model(
            source_path,
            model_format,
            model_framework,
        ):
            return ModelLoader._load_serialized_model(
                source_path,
                uri=resolved["uri"],
                model_format=model_format,
                model_framework=model_framework,
                model_type=model_type,
                model_entrypoint=model_entrypoint,
            )

        if model_format == "model_directory":
            raise ValueError(
                "Unsupported model_directory contract. "
                f"uri={resolved['uri']!r}, resolved_path={str(artifact_path)!r}, "
                f"entrypoint={model_entrypoint!r}, "
                f"source_path={str(source_path)!r}, "
                f"model_framework={model_framework!r}"
            )

        if model_format == "onnx" or model_framework == "onnxruntime":
            raise NotImplementedError(
                "ONNX loader is not implemented yet. "
                f"uri={resolved['uri']!r}, resolved_path={str(source_path)!r}"
            )

        raise ValueError(
            "Unsupported model metadata for loading a model artifact. "
            f"uri={resolved['uri']!r}, resolved_path={str(source_path)!r}, "
            f"model_format={model_format!r}, "
            f"model_framework={model_framework!r}, "
            f"model_type={model_type!r}, "
            f"model_entrypoint={model_entrypoint!r}"
        )

    @staticmethod
    def load_tokenizer(audit_context):
        """
        Load a HuggingFace tokenizer when the artifact metadata indicates an
        LLM / transformer stack. Return None for non-LLM models.
        """
        resolved = ModelLoader._resolve_artifact(audit_context)
        model_format = resolved["model_format"]
        model_framework = resolved["model_framework"]

        if not ModelLoader._is_huggingface_stack(model_format, model_framework):
            return None

        source_path = resolved["source_path"]
        allow_soft_failure = ModelLoader._is_metadata_only_huggingface_source(
            source_path
        )
        if not source_path.exists():
            if allow_soft_failure:
                print(
                    "[ModelLoader] Tokenizer source path is missing for a "
                    "metadata-only LLM artifact; continuing without tokenizer. "
                    f"uri={resolved['uri']!r}, resolved_path={str(source_path)!r}"
                )
                return None
            raise FileNotFoundError(
                "Tokenizer source path not found. "
                f"uri={resolved['uri']!r}, resolved_path={str(source_path)!r}"
            )

        return ModelLoader._load_huggingface_tokenizer(
            source_path,
            uri=resolved["uri"],
            model_format=model_format,
            model_framework=model_framework,
            allow_soft_failure=allow_soft_failure,
        )

    @staticmethod
    def load_metadata(audit_context) -> dict[str, Any]:
        """
        Return a normalized metadata dictionary for the resolved artifact.
        """
        resolved = ModelLoader._resolve_artifact(audit_context)
        metadata: dict[str, Any] = {
            "artifact_uri": resolved["uri"],
            "resolved_artifact_path": str(resolved["artifact_path"]),
            "source_path": str(resolved["source_path"]),
            "model_format": resolved["model_format"],
            "model_framework": resolved["model_framework"],
            "model_type": resolved["model_type"],
            "model_entrypoint": resolved["model_entrypoint"],
            "is_directory_contract": resolved["is_directory_contract"],
            "loaded_metadata_files": [],
            "directory_metadata": {},
            "status": "loaded",
            "is_loadable": True,
            "reason": None,
        }

        if resolved["is_directory_contract"] or resolved["source_path"].is_dir():
            directory_metadata = ModelLoader._collect_directory_metadata(
                resolved["artifact_path"],
                resolved["source_path"],
            )
            metadata["directory_metadata"] = directory_metadata["files"]
            metadata["loaded_metadata_files"] = directory_metadata["loaded_files"]

            if directory_metadata["loaded_files"]:
                print(
                    "[ModelLoader] Loaded metadata files: "
                    f"{directory_metadata['loaded_files']}"
                )

            if ModelLoader._is_huggingface_stack(
                resolved["model_format"],
                resolved["model_framework"],
            ) and ModelLoader._is_metadata_only_huggingface_source(
                resolved["source_path"]
            ):
                metadata["status"] = "metadata_only"
                metadata["is_loadable"] = False
                metadata["reason"] = (
                    "The provided LLM artifact contains configuration "
                    "metadata but no loadable model weights."
                )
        return metadata

    @staticmethod
    def _normalize(value: Any) -> str:
        return str(value).strip().lower() if value is not None else ""

    @staticmethod
    def _resolve_artifact(audit_context) -> dict[str, Any]:
        uri = getattr(audit_context, "model_artifact_uri", None)
        if not uri:
            raise ValueError(
                "audit_context.model_artifact_uri is required to load a model."
            )

        artifact_path = resolve_artifact_path(uri)
        model_format = ModelLoader._normalize(
            getattr(audit_context, "model_format", None)
        )
        model_framework = ModelLoader._normalize(
            getattr(audit_context, "model_framework", None)
        )
        model_type = getattr(audit_context, "model_type", None) or None
        model_entrypoint = getattr(audit_context, "model_entrypoint", None) or None
        system_type = ModelLoader._normalize(
            getattr(audit_context, "system_type", "")
        )

        if not model_format and not model_framework:
            if system_type in {"llm", "agentic"}:
                model_format = "huggingface"
                model_framework = "transformers"
            else:
                model_format = "joblib"
                model_framework = "sklearn"

        if not model_format:
            model_format = ModelLoader._infer_format_from_framework(
                model_framework
            )

        if not model_framework:
            model_framework = ModelLoader._infer_framework_from_format(
                model_format
            )

        is_directory_contract = model_format == "model_directory"
        if is_directory_contract:
            if not artifact_path.exists():
                raise FileNotFoundError(
                    "Model directory artifact not found. "
                    f"uri={uri!r}, resolved_path={str(artifact_path)!r}"
                )
            if not artifact_path.is_dir():
                raise NotADirectoryError(
                    "Model directory contract requires a directory. "
                    f"uri={uri!r}, resolved_path={str(artifact_path)!r}"
                )
            if not model_entrypoint:
                raise ValueError(
                    "audit_context.model_entrypoint is required when "
                    "model_format == 'model_directory'. "
                    f"uri={uri!r}, resolved_path={str(artifact_path)!r}"
                )

            source_path = artifact_path / model_entrypoint
        else:
            source_path = artifact_path

        return {
            "uri": uri,
            "artifact_path": artifact_path,
            "source_path": source_path,
            "model_format": model_format,
            "model_framework": model_framework,
            "model_type": model_type,
            "model_entrypoint": model_entrypoint,
            "is_directory_contract": is_directory_contract,
        }

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
    def _find_huggingface_weight_files(source: Path) -> list[Path]:
        files = list(source.rglob("*.safetensors"))
        files.extend(source.rglob("*.bin"))
        return sorted(set(files))

    @staticmethod
    def _collect_huggingface_metadata_files(source: Path) -> list[str]:
        if not source.exists():
            return []

        metadata_files: list[str] = []
        for pattern in ("*.json",):
            for file_path in sorted(source.rglob(pattern)):
                metadata_files.append(str(file_path.relative_to(source)))
        return metadata_files

    @staticmethod
    def _is_metadata_only_huggingface_source(source: Path) -> bool:
        if not source.exists() or not source.is_dir():
            return False
        return len(ModelLoader._find_huggingface_weight_files(source)) == 0

    @staticmethod
    def _should_load_serialized_model(
        source_path: Path,
        model_format: str,
        model_framework: str,
    ) -> bool:
        if source_path.is_file():
            return (
                source_path.suffix.lower() in {".joblib", ".pkl", ".pickle"}
                or ModelLoader._is_serialized_stack(model_format, model_framework)
            )

        return False

    @staticmethod
    def _load_serialized_model(
        path: Path,
        uri: str | None = None,
        model_format: str | None = None,
        model_framework: str | None = None,
        model_type: str | None = None,
        model_entrypoint: str | None = None,
    ):
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
                        f"model_type={model_type!r}, "
                        f"model_entrypoint={model_entrypoint!r}, "
                        f"joblib_error={type(joblib_error).__name__}: {joblib_error}, "
                        f"pickle_error={type(pickle_error).__name__}: {pickle_error}"
                    ) from pickle_exc

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

        wrapped_bundle = ModelLoader._wrap_encoded_sklearn_bundle(
            loaded,
            uri=uri,
            path=path,
            model_format=model_format,
            model_framework=model_framework,
            model_type=model_type,
            model_entrypoint=model_entrypoint,
        )
        final_model = wrapped_bundle if wrapped_bundle is not None else unwrap_artifact(loaded)
        if load_warnings:
            try:
                setattr(final_model, "_uagf_load_warnings", load_warnings)
            except Exception:
                # Report metadata is best-effort and must not alter model loading.
                pass
        return final_model

    @staticmethod
    def _wrap_encoded_sklearn_bundle(
        artifact,
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

    @staticmethod
    def _load_huggingface_model(
        path: Path,
        uri: str | None = None,
        model_format: str | None = None,
        model_framework: str | None = None,
        model_type: str | None = None,
        model_entrypoint: str | None = None,
    ):
        source = path if path.is_dir() else path.parent
        if not source.exists():
            raise FileNotFoundError(
                "HuggingFace model source directory not found. "
                f"uri={uri!r}, resolved_path={str(path)!r}, "
                f"source_dir={str(source)!r}, "
                f"model_format={model_format!r}, "
                f"model_framework={model_framework!r}, "
                f"model_type={model_type!r}, "
                f"model_entrypoint={model_entrypoint!r}"
            )

        weight_files = ModelLoader._find_huggingface_weight_files(source)
        if not weight_files:
            metadata_files = ModelLoader._collect_huggingface_metadata_files(
                source
            )
            print(
                "[ModelLoader] LLM model directory is metadata-only; no "
                "loadable weights found. LLM inference will be skipped. "
                f"uri={uri!r}, resolved_path={str(path)!r}, "
                f"source_dir={str(source)!r}"
            )
            return MetadataOnlyLLMArtifact(
                resolved_path=str(path),
                model_type=model_type,
                model_framework=model_framework,
                model_entrypoint=model_entrypoint,
                loaded_metadata_files=metadata_files,
                metadata={
                    "artifact_uri": uri,
                    "resolved_artifact_path": str(path),
                    "source_path": str(source),
                    "model_format": model_format,
                    "model_framework": model_framework,
                    "model_type": model_type,
                    "model_entrypoint": model_entrypoint,
                    "loaded_metadata_files": metadata_files,
                    "status": "metadata_only",
                    "is_loadable": False,
                    "reason": (
                        "The provided LLM artifact contains configuration "
                        "metadata but no loadable model weights."
                    ),
                },
            )

        try:
            from transformers import AutoModelForCausalLM
            from transformers import AutoTokenizer
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required to load HuggingFace models. "
                f"uri={uri!r}, resolved_path={str(path)!r}, "
                f"model_format={model_format!r}, "
                f"model_framework={model_framework!r}, "
                f"model_type={model_type!r}, "
                f"model_entrypoint={model_entrypoint!r}"
            ) from exc

        print(f"[ModelLoader] HuggingFace source directory: {str(source)!r}")
        try:
            model = AutoModelForCausalLM.from_pretrained(str(source))
            tokenizer = AutoTokenizer.from_pretrained(str(source))
            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load HuggingFace model artifact. "
                f"uri={uri!r}, resolved_path={str(path)!r}, "
                f"source_dir={str(source)!r}, "
                f"model_format={model_format!r}, "
                f"model_framework={model_framework!r}, "
                f"model_type={model_type!r}, "
                f"model_entrypoint={model_entrypoint!r}, "
                f"error={type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _load_huggingface_tokenizer(
        path: Path,
        uri: str | None = None,
        model_format: str | None = None,
        model_framework: str | None = None,
        allow_soft_failure: bool = False,
    ):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            if allow_soft_failure:
                print(
                    "[ModelLoader] transformers is unavailable for a "
                    "metadata-only LLM artifact; continuing without tokenizer. "
                    f"uri={uri!r}, resolved_path={str(path)!r}, "
                    f"model_format={model_format!r}, "
                    f"model_framework={model_framework!r}"
                )
                return None
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

        print(f"[ModelLoader] HuggingFace tokenizer source: {str(source)!r}")
        try:
            return AutoTokenizer.from_pretrained(str(source))
        except Exception as exc:
            if allow_soft_failure:
                print(
                    "[ModelLoader] Tokenizer loading failed for a "
                    "metadata-only LLM artifact; continuing without tokenizer. "
                    f"uri={uri!r}, resolved_path={str(path)!r}, "
                    f"source_dir={str(source)!r}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                return None
            raise RuntimeError(
                "Failed to load HuggingFace tokenizer artifact. "
                f"uri={uri!r}, resolved_path={str(path)!r}, "
                f"source_dir={str(source)!r}, "
                f"model_format={model_format!r}, "
                f"model_framework={model_framework!r}, "
                f"error={type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _collect_directory_metadata(
        base_dir: Path,
        source_dir: Path,
    ) -> dict[str, Any]:
        loaded_files: list[str] = []
        metadata_files: dict[str, Any] = {}

        if not base_dir.exists() or not base_dir.is_dir():
            return {
                "files": metadata_files,
                "loaded_files": loaded_files,
            }

        for json_file in sorted(base_dir.rglob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                payload = {
                    "_error": f"{type(exc).__name__}: {exc}",
                }

            key = f"model_dir:{json_file.relative_to(base_dir).as_posix()}"
            if key not in metadata_files:
                metadata_files[key] = payload
                loaded_files.append(key)

        return {
            "files": metadata_files,
            "loaded_files": loaded_files,
        }
