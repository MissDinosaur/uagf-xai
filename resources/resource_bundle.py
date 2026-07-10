"""
Resource bundle.

This module defines a unified container for all resources
required by the UAGF-XAI evidence generation pipeline.
"""

from dataclasses import dataclass
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

    training_dataset: pd.DataFrame

    evaluation_dataset: pd.DataFrame

    # ---------- LLM ----------

    tokenizer: Any | None = None

    embedding_model: Any | None = None

    prompt_template: str | None = None

    golden_dataset: pd.DataFrame | None = None