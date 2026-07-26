"""
Dataset and auxiliary resource loading utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .artifact_utils import resolve_artifact_path


class DatasetLoader:
    """
    Load datasets and auxiliary LLM resources from resolved local paths.
    """

    @staticmethod
    def _require_uri(uri: str | None, resource_name: str) -> str:
        if not uri:
            raise ValueError(f"audit_context.{resource_name} is required.")
        return uri

    @staticmethod
    def _read_json(path: Path) -> Any:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _load_tabular_csv(uri: str, label: str):
        local = resolve_artifact_path(uri)
        if not local.exists():
            raise FileNotFoundError(
                f"{label} not found: uri={uri!r}, resolved_path={str(local)!r}"
            )
        df = pd.read_csv(local)
        print(f"[DatasetLoader] Loaded {label}: {str(local)!r}")
        return df

    @staticmethod
    def _load_structured_resource(uri: str, label: str) -> Any:
        local = resolve_artifact_path(uri)
        if not local.exists():
            raise FileNotFoundError(
                f"{label} not found: uri={uri!r}, resolved_path={str(local)!r}"
            )

        suffix = local.suffix.lower()
        if suffix == ".json":
            value = DatasetLoader._read_json(local)
        elif suffix == ".jsonl":
            with open(local, encoding="utf-8") as f:
                value = [json.loads(line) for line in f if line.strip()]
        elif suffix in {".txt", ".md"}:
            value = DatasetLoader._read_text(local)
        elif suffix == ".csv":
            value = pd.read_csv(local)
        else:
            value = DatasetLoader._read_text(local)

        print(f"[DatasetLoader] Loaded {label}: {str(local)!r}")
        return value

    @staticmethod
    def load_training(audit_context):
        uri = DatasetLoader._require_uri(
            getattr(audit_context, "training_dataset_uri", None),
            "training_dataset_uri",
        )
        return DatasetLoader._load_tabular_csv(uri, "training dataset")

    @staticmethod
    def load_evaluation(audit_context):
        uri = DatasetLoader._require_uri(
            getattr(audit_context, "evaluation_dataset_uri", None),
            "evaluation_dataset_uri",
        )
        return DatasetLoader._load_tabular_csv(uri, "evaluation dataset")

    @staticmethod
    def load_golden_set(audit_context):
        uri = getattr(audit_context, "golden_set_uri", None)
        if not uri:
            return None
        return DatasetLoader._load_structured_resource(uri, "golden set")

    @staticmethod
    def load_system_prompt(audit_context):
        uri = getattr(audit_context, "system_prompt_uri", None)
        if not uri:
            return None
        return DatasetLoader._load_structured_resource(uri, "system prompt")

    @staticmethod
    def load_rag_manifest(audit_context):
        uri = getattr(audit_context, "rag_manifest_uri", None)
        if not uri:
            return None
        return DatasetLoader._load_structured_resource(uri, "RAG manifest")

    @staticmethod
    def load_guardrail_config(audit_context):
        uri = getattr(audit_context, "guardrail_config_uri", None)
        if not uri:
            return None
        return DatasetLoader._load_structured_resource(uri, "guardrail config")

    @staticmethod
    def load_traditional(audit_context):
        train = DatasetLoader.load_training(audit_context)
        eval_ = DatasetLoader.load_evaluation(audit_context)
        return train, eval_

