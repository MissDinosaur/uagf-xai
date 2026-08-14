"""Offline loading for S6-owned LLM evaluation embedding instruments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifact_utils import resolve_artifact_path


@dataclass
class LocalSentenceEmbeddingModel:
    """Plain-Transformers sentence encoder with masked mean pooling."""

    model: Any
    tokenizer: Any
    resolved_path: str
    model_name: str
    device: str = "cpu"

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        if not texts:
            return np.empty((0, 0), dtype=float)
        encoded = self.tokenizer(
            [str(text) for text in texts],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            token_embeddings = self.model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())
        mask = mask.to(token_embeddings.dtype)
        pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        normalized = functional.normalize(pooled, p=2, dim=1)
        return normalized.cpu().numpy()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "evaluation_embedding_model": self.model_name,
            "resolved_path": self.resolved_path,
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "pooling": "attention_mask_mean_pooling",
            "normalization": "l2",
            "device": self.device,
            "local_files_only": True,
        }


class LLMEvaluationModelLoader:
    """Load the local S6 evaluation model without network access."""

    @staticmethod
    def load(evidence_config) -> LocalSentenceEmbeddingModel | None:
        if evidence_config is None:
            return None
        uri = evidence_config.evaluation_embedding_model_uri
        if not uri:
            return None
        source = resolve_artifact_path(uri)
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError(
                "Evaluation embedding model directory not found. "
                f"uri={uri!r}, resolved_path={str(source)!r}"
            )
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required for LLM evaluation embeddings."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            str(source), local_files_only=True
        )
        model = AutoModel.from_pretrained(str(source), local_files_only=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        name = evidence_config.evaluation_embedding_model
        print(f"[ResourceLoader] Loaded local evaluation embedding model: {source}")
        return LocalSentenceEmbeddingModel(
            model=model,
            tokenizer=tokenizer,
            resolved_path=str(source),
            model_name=name,
            device=str(device),
        )
