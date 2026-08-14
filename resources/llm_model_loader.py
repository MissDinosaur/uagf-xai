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


@dataclass
class LocalCausalLMArtifact:
    """Offline wrapper around an S5-referenced fitted causal language model."""

    model: Any
    tokenizer: Any
    resolved_path: str
    model_type: str | None = None
    model_framework: str | None = None
    model_entrypoint: str | None = None
    loaded_metadata_files: list[str] = field(default_factory=list)
    device: str = "cpu"
    status: str = "loaded"
    is_loadable: bool = True

    def generate_response(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        """Generate and return only newly generated continuation text."""
        return self.generate_response_with_metadata(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )["text"]

    def generate_response_with_metadata(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate a token-sliced continuation and auditable call metadata."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        generation = {
            "max_new_tokens": int(max_new_tokens),
            "min_new_tokens": min(4, int(max_new_tokens)),
            "do_sample": bool(do_sample),
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generation["temperature"] = float(temperature or 0.8)
            generation["top_p"] = float(top_p or 0.9)

        devices = [self.model.device.index] if self.model.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
            with torch.inference_mode():
                output = self.model.generate(**inputs, **generation)

        input_token_count = int(inputs["input_ids"].shape[1])
        if output.ndim != 2 or output.shape[0] != 1:
            raise RuntimeError(
                "The local causal LM must return one two-dimensional token sequence."
            )
        if output.shape[1] < input_token_count:
            raise RuntimeError(
                "The local causal LM returned fewer tokens than the supplied prompt."
            )
        continuation_ids = output[0, input_token_count:]
        text = self.tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True,
        ).strip()
        return {
            "text": text,
            "continuation_only": True,
            "input_token_count": input_token_count,
            "generated_token_count": int(continuation_ids.shape[0]),
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "seed": seed,
            "temperature": generation.get("temperature"),
            "top_p": generation.get("top_p"),
        }

    def __call__(self, prompt: str, **kwargs):
        """Provide the narrow Transformers-pipeline interface used by legacy callers."""
        count = int(kwargs.pop("num_return_sequences", 1))
        base_seed = kwargs.pop("seed", None)
        return_full_text = bool(kwargs.pop("return_full_text", True))
        outputs = []
        for index in range(count):
            seed = None if base_seed is None else int(base_seed) + index
            generated = self.generate_response(prompt, seed=seed, **kwargs)
            text = f"{prompt}{generated}" if return_full_text else generated
            outputs.append({"generated_text": text})
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "is_loadable": self.is_loadable,
            "resolved_artifact_path": self.resolved_path,
            "model_type": self.model_type,
            "model_framework": self.model_framework,
            "model_entrypoint": self.model_entrypoint,
            "loaded_metadata_files": list(self.loaded_metadata_files),
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "device": self.device,
            "local_files_only": True,
        }


class LLMModelLoader:
    """Load local HuggingFace LLM artifacts and their tokenizers."""

    LLM_FORMATS = frozenset(
        {"huggingface", "hf", "huggingface_pretrained", "huggingface_adapter"}
    )
    LLM_FRAMEWORKS = frozenset(
        {"huggingface", "transformers", "huggingface_transformers"}
    )

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
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
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
            tokenizer = AutoTokenizer.from_pretrained(
                str(source), local_files_only=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(source), local_files_only=True
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            return LocalCausalLMArtifact(
                model=model,
                tokenizer=tokenizer,
                resolved_path=str(source),
                model_type=model_type,
                model_framework=model_framework,
                model_entrypoint=model_entrypoint,
                loaded_metadata_files=cls.collect_metadata_files(source),
                device=str(device),
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
            return AutoTokenizer.from_pretrained(
                str(source), local_files_only=True
            )
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
