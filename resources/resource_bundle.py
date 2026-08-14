"""
Resource bundle.

This module defines a unified container for all resources
required by the UAGF-XAI evidence generation pipeline.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ResourceBundle:
    """
    Container for resources loaded from S5.

    Attributes
    ----------
    model
        Loaded machine learning model.

    training_dataset
        Training dataset.

    evaluation_dataset
        Evaluation dataset.
    """
    # ---------- Traditional ML ----------

    model: Any

    model_metadata: dict[str, Any] = field(default_factory=dict)

    model_feature_columns: list[str] | None = None

    model_feature_scope_source: str | None = None

    training_dataset: pd.DataFrame | None = None

    evaluation_dataset: pd.DataFrame | None = None

    # ---------- LLM ----------

    tokenizer: Any | None = None

    embedding_model: Any | None = None

    evaluation_embedding_metadata: dict[str, Any] = field(default_factory=dict)

    prompt_template: str | None = None

    golden_dataset: Any | None = None

    system_prompt: str | None = None

    rag_manifest: Any | None = None

    guardrail_config: Any | None = None

    semantic_drift_dataset: Any | None = None

    fairness_prompt_pairs: Any | None = None

    llm_evidence_config: Any | None = None
