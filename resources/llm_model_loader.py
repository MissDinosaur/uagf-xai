"""LLM and Agentic model loading for local HuggingFace artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MetadataOnlyLLMArtifact:
    """Describe an LLM artifact that has metadata but no executable weights."""

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


class LLMModelLoader:
    """Load local HuggingFace LLM artifacts and their tokenizers."""

    LLM_FORMATS = frozenset({"huggingface", "hf"})
    LLM_FRAMEWORKS = frozenset({"huggingface", "transformers"})

    @classmethod
    def supports(cls, model_format: str, model_framework: str) -> bool:
        return (
            model_format in cls.LLM_FORMATS
            or model_framework in cls.LLM_FRAMEWORKS
        )

    @staticmethod
    def find_weight_files(source: Path) -> list[Path]:
        files = list(source.rglob("*.safetensors"))
        files.extend(source.rglob("*.bin"))
        return sorted(set(files))

    @staticmethod
    def collect_metadata_files(source: Path) -> list[str]:
        if not source.exists():
            return []
        return [
            str(file_path.relative_to(source))
            for file_path in sorted(source.rglob("*.json"))
        ]

    @classmethod
    def is_metadata_only(cls, source: Path) -> bool:
        if not source.exists() or not source.is_dir():
            return False
        return len(cls.find_weight_files(source)) == 0

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
        """Load an LLM pipeline or return a metadata-only artifact wrapper."""
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

        weight_files = cls.find_weight_files(source)
        if not weight_files:
            return cls._metadata_only_artifact(
                path,
                source,
                uri=uri,
                model_format=model_format,
                model_framework=model_framework,
                model_type=model_type,
                model_entrypoint=model_entrypoint,
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
            return pipeline("text-generation", model=model, tokenizer=tokenizer)
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

    @classmethod
    def _metadata_only_artifact(
        cls,
        path: Path,
        source: Path,
        *,
        uri: str | None,
        model_format: str | None,
        model_framework: str | None,
        model_type: str | None,
        model_entrypoint: str | None,
    ) -> MetadataOnlyLLMArtifact:
        metadata_files = cls.collect_metadata_files(source)
        reason = (
            "The provided LLM artifact contains configuration metadata but no "
            "loadable model weights."
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
                "reason": reason,
            },
        )

    @staticmethod
    def load_tokenizer(
        path: Path,
        *,
        uri: str | None = None,
        model_format: str | None = None,
        model_framework: str | None = None,
        allow_soft_failure: bool = False,
    ):
        """Load a HuggingFace tokenizer with metadata-only soft failure support."""
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
