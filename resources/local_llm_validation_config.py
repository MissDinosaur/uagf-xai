"""S6-owned configuration for local LLM evidence execution validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from adapters.s5_audit_adapter import AuditContext


@dataclass(frozen=True)
class S6LLMEvidenceConfig:
    """Resources selected by S6 to execute local LLM evidence methods."""

    evaluation_embedding_model: str
    evaluation_embedding_model_uri: str
    semantic_drift_dataset_uri: str
    fairness_prompt_pairs_uri: str

    @classmethod
    def from_dict(cls, data: dict) -> "S6LLMEvidenceConfig":
        missing = [
            field
            for field in cls.__dataclass_fields__
            if not str(data.get(field) or "").strip()
        ]
        if missing:
            raise ValueError(
                "llm_evidence_config is missing required fields: "
                f"{sorted(missing)}"
            )
        return cls(**{field: str(data[field]) for field in cls.__dataclass_fields__})


@dataclass(frozen=True)
class LocalLLMValidationConfig:
    """Complete S6 local case without representing it as an S5 submission."""

    audit_context: AuditContext
    llm_evidence_config: S6LLMEvidenceConfig
    source_path: str

    @classmethod
    def load(cls, path: str | Path) -> "LocalLLMValidationConfig":
        source = Path(path)
        with open(source, encoding="utf-8") as config_file:
            data = json.load(config_file)
        audit_data = data.get("audit_context")
        evidence_data = data.get("llm_evidence_config")
        if not isinstance(audit_data, dict) or not isinstance(evidence_data, dict):
            raise ValueError(
                "Local LLM validation config requires audit_context and "
                "llm_evidence_config objects."
            )
        audit_context = AuditContext(**audit_data)
        if audit_context.system_type not in {"llm", "agentic"}:
            raise ValueError("Local LLM validation audit_context must describe an LLM.")
        if audit_context.task_type != "llm_generation":
            raise ValueError("Local LLM validation task_type must be llm_generation.")
        return cls(
            audit_context=audit_context,
            llm_evidence_config=S6LLMEvidenceConfig.from_dict(evidence_data),
            source_path=source.as_posix(),
        )
