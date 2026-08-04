"""Facade for resolving and dispatching S5-provided model artifacts.

Traditional ML serialization details live in ``traditional_model_loader``.
LLM and HuggingFace details live in ``llm_model_loader``. This module keeps
the stable ``ModelLoader`` API used by ``ResourceLoader`` and existing users.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifact_utils import resolve_artifact_path
from .llm_model_loader import LLMModelLoader, MetadataOnlyLLMArtifact
from .traditional_model_loader import EncodedSklearnModel, TraditionalModelLoader

__all__ = [
    "EncodedSklearnModel",
    "MetadataOnlyLLMArtifact",
    "ModelLoader",
]


class ModelLoader:
    """Resolve the S5 model contract and delegate to the correct loader."""

    @staticmethod
    def load(audit_context):
        resolved = ModelLoader._resolve_artifact(audit_context)
        artifact_path = resolved["artifact_path"]
        source_path = resolved["source_path"]
        model_format = resolved["model_format"]
        model_framework = resolved["model_framework"]
        model_type = resolved["model_type"]
        model_entrypoint = resolved["model_entrypoint"]

        if not source_path.exists():
            raise FileNotFoundError(
                "Model artifact not found. "
                f"uri={resolved['uri']!r}, resolved_path={str(source_path)!r}, "
                f"model_artifact_kind={resolved['model_artifact_kind']!r}, "
                f"model_format={model_format!r}, "
                f"model_framework={model_framework!r}, "
                f"model_type={model_type!r}, "
                f"model_entrypoint={model_entrypoint!r}"
            )

        ModelLoader._log_resolved_artifact(resolved)

        # If it belongs to LLM/Agentic case, then its model path should be a directory
        if LLMModelLoader.supports(model_format, model_framework):
            if source_path.is_dir():
                return LLMModelLoader.load(
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
                    f"uri={resolved['uri']!r}, "
                    f"resolved_path={str(artifact_path)!r}, "
                    f"entrypoint={model_entrypoint!r}, "
                    f"source_path={str(source_path)!r}"
                )

        if TraditionalModelLoader.supports(source_path, model_format, model_framework):
            return TraditionalModelLoader.load(
                source_path,
                uri=resolved["uri"],
                model_format=model_format,
                model_framework=model_framework,
                model_type=model_type,
                model_entrypoint=model_entrypoint,
            )

        if resolved["is_directory_contract"]:
            raise ValueError(
                "Unsupported model directory contract. "
                f"uri={resolved['uri']!r}, "
                f"resolved_path={str(artifact_path)!r}, "
                f"entrypoint={model_entrypoint!r}, "
                f"source_path={str(source_path)!r}, "
                f"model_framework={model_framework!r}"
            )
        if model_format == "onnx" or model_framework == "onnxruntime":
            raise NotImplementedError(
                "ONNX loader is not implemented yet. "
                f"uri={resolved['uri']!r}, "
                f"resolved_path={str(source_path)!r}"
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
        """Load a tokenizer only for an LLM/HuggingFace model contract."""
        resolved = ModelLoader._resolve_artifact(audit_context)
        model_format = resolved["model_format"]
        model_framework = resolved["model_framework"]
        if not LLMModelLoader.supports(model_format, model_framework):
            return None

        source_path = resolved["source_path"]
        allow_soft_failure = LLMModelLoader.is_metadata_only(source_path)
        if not source_path.exists():
            if allow_soft_failure:
                print(
                    "[ModelLoader] Tokenizer source path is missing for a "
                    "metadata-only LLM artifact; continuing without tokenizer. "
                    f"uri={resolved['uri']!r}, "
                    f"resolved_path={str(source_path)!r}"
                )
                return None
            raise FileNotFoundError(
                "Tokenizer source path not found. "
                f"uri={resolved['uri']!r}, "
                f"resolved_path={str(source_path)!r}"
            )
        return LLMModelLoader.load_tokenizer(
            source_path,
            uri=resolved["uri"],
            model_format=model_format,
            model_framework=model_framework,
            allow_soft_failure=allow_soft_failure,
        )

    @staticmethod
    def load_metadata(audit_context) -> dict[str, Any]:
        """Return normalized metadata for the resolved model artifact."""
        resolved = ModelLoader._resolve_artifact(audit_context)
        metadata: dict[str, Any] = {
            "artifact_uri": resolved["uri"],
            "resolved_artifact_path": str(resolved["artifact_path"]),
            "source_path": str(resolved["source_path"]),
            "model_artifact_kind": resolved["model_artifact_kind"],
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
                resolved["artifact_path"]
            )
            metadata["directory_metadata"] = directory_metadata["files"]
            metadata["loaded_metadata_files"] = directory_metadata["loaded_files"]
            if directory_metadata["loaded_files"]:
                print(
                    "[ModelLoader] Loaded metadata files: "
                    f"{directory_metadata['loaded_files']}"
                )

            if (
                LLMModelLoader.supports(
                    resolved["model_format"],
                    resolved["model_framework"],
                )
                and LLMModelLoader.is_metadata_only(resolved["source_path"])
            ):
                metadata["status"] = "metadata_only"
                metadata["is_loadable"] = False
                metadata["reason"] = (
                    "The provided LLM artifact contains configuration metadata "
                    "but no loadable model weights."
                )
        return metadata

    @staticmethod
    def _log_resolved_artifact(resolved: dict[str, Any]) -> None:
        if resolved["is_directory_contract"]:
            print(
                "[ModelLoader] Loaded model directory: "
                f"uri={resolved['uri']!r}, "
                f"resolved_path={str(resolved['artifact_path'])!r}"
            )
            print(
                "[ModelLoader] Model entrypoint: "
                f"{resolved['model_entrypoint']!r} -> "
                f"{str(resolved['source_path'])!r}"
            )
        else:
            print(
                "[ModelLoader] Loaded single-file model artifact: "
                f"uri={resolved['uri']!r}, "
                f"resolved_path={str(resolved['source_path'])!r}"
            )

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
        model_artifact_kind = ModelLoader._normalize(
            getattr(audit_context, "model_artifact_kind", None)
        )
        system_type = ModelLoader._normalize(
            getattr(audit_context, "system_type", "")
        )

        if not model_artifact_kind:
            raise ValueError(
                "audit_context.model_artifact_kind is required and must be "
                "'single_file' or 'directory'."
            )
        if model_artifact_kind not in {"single_file", "directory"}:
            raise ValueError(
                "Unsupported audit_context.model_artifact_kind. "
                f"Expected 'single_file' or 'directory', got "
                f"{model_artifact_kind!r}."
            )

        if not model_format and not model_framework:
            if system_type in {"llm", "agentic"}:
                model_format = "huggingface"
                model_framework = "transformers"
            else:
                model_format = "joblib"
                model_framework = "sklearn"
        if not model_format:
            model_format = ModelLoader._infer_model_format_from_framework(model_framework)
        if not model_framework:
            model_framework = ModelLoader._infer_model_framework_from_format(model_format)

        is_directory_contract = model_artifact_kind == "directory"
        if is_directory_contract:
            ModelLoader._validate_directory_contract(uri, artifact_path, model_entrypoint)
            source_path = artifact_path / model_entrypoint
        else:
            source_path = artifact_path
            if artifact_path.exists() and artifact_path.is_dir():
                raise IsADirectoryError(
                    "Single-file model artifact contract resolved to a directory. "
                    f"uri={uri!r}, resolved_path={str(artifact_path)!r}"
                )

        return {
            "uri": uri,
            "artifact_path": artifact_path,
            "source_path": source_path,
            "model_artifact_kind": model_artifact_kind,
            "model_format": model_format,
            "model_framework": model_framework,
            "model_type": model_type,
            "model_entrypoint": model_entrypoint,
            "is_directory_contract": is_directory_contract,
        }

    @staticmethod
    def _validate_directory_contract(
        uri: str,
        artifact_path: Path,
        model_entrypoint: str | None,
    ) -> None:
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
                "model_artifact_kind == 'directory'. "
                f"uri={uri!r}, resolved_path={str(artifact_path)!r}"
            )

    @staticmethod
    def _infer_model_format_from_framework(model_framework: str) -> str:
        if model_framework in {"sklearn", "scikit-learn", "xgboost", "lightgbm"}:
            return "joblib"
        if model_framework in {"transformers", "huggingface"}:
            return "huggingface"
        if model_framework == "onnxruntime":
            return "onnx"
        return model_framework

    @staticmethod
    def _infer_model_framework_from_format(model_format: str) -> str:
        if model_format in {"joblib", "pkl", "pickle"}:
            return "sklearn"
        if model_format in {"huggingface", "hf"}:
            return "transformers"
        if model_format == "onnx":
            return "onnxruntime"
        return model_format

    @staticmethod
    def _collect_directory_metadata(base_dir: Path) -> dict[str, Any]:
        loaded_files: list[str] = []
        metadata_files: dict[str, Any] = {}
        if not base_dir.exists() or not base_dir.is_dir():
            return {"files": metadata_files, "loaded_files": loaded_files}

        for json_file in sorted(base_dir.rglob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as metadata_file:
                    payload = json.load(metadata_file)
            except Exception as exc:
                payload = {"_error": f"{type(exc).__name__}: {exc}"}
            key = f"model_dir:{json_file.relative_to(base_dir).as_posix()}"
            metadata_files[key] = payload
            loaded_files.append(key)
        return {"files": metadata_files, "loaded_files": loaded_files}
