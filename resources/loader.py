"""
Unified resource loader.
"""

from __future__ import annotations

from .model_loader import ModelLoader
from .dataset_loader import DatasetLoader
from .resource_bundle import ResourceBundle


class ResourceLoader:
    """
    Facade for loading all resources required by S6.
    """

    @staticmethod
    def load_model(audit_context):
        return ModelLoader.load(audit_context)

    @staticmethod
    def load_tokenizer(audit_context):
        return ModelLoader.load_tokenizer(audit_context)

    @staticmethod
    def load_model_metadata(audit_context):
        return ModelLoader.load_metadata(audit_context)

    @staticmethod
    def _is_llm_contract(audit_context) -> bool:
        system_type = str(getattr(audit_context, "system_type", "") or "").strip().lower()
        return system_type in {"llm", "agentic"}

    @staticmethod
    def load_traditional_resources(audit_context):
        return DatasetLoader.load_traditional(audit_context)

    @staticmethod
    def load_llm_resources(audit_context):
        return {
            "golden_set": DatasetLoader.load_golden_set(audit_context),
            "system_prompt": DatasetLoader.load_system_prompt(audit_context),
            "rag_manifest": DatasetLoader.load_rag_manifest(audit_context),
            "guardrail_config": DatasetLoader.load_guardrail_config(audit_context),
        }

    @staticmethod
    def load_bundle(audit_context) -> ResourceBundle:
        if ResourceLoader._is_llm_contract(audit_context):
            print("[ResourceLoader] LLM golden-set resource contract detected.")
        else:
            print("[ResourceLoader] Traditional ML resource contract detected.")

        model = ResourceLoader.load_model(audit_context)
        tokenizer = ResourceLoader.load_tokenizer(audit_context)
        model_metadata = ResourceLoader.load_model_metadata(audit_context)
        model_feature_columns = getattr(model, "model_feature_columns", None)
        if model_feature_columns is None:
            fitted_names = getattr(model, "feature_names_in_", None)
            model_feature_columns = list(fitted_names) if fitted_names is not None else None
        if model_feature_columns is not None:
            model_feature_columns = list(model_feature_columns)
        for key in (
            "input_adapter",
            "text_feature_column",
            "vectorizer_class",
            "estimator_class",
            "vocabulary_size",
        ):
            value = getattr(model, key, None)
            if value is not None:
                model_metadata[key] = value

        if ResourceLoader._is_llm_contract(audit_context):
            llm_resources = ResourceLoader.load_llm_resources(audit_context)
            return ResourceBundle(
                model=model,
                model_metadata=model_metadata,
                model_feature_columns=model_feature_columns,
                model_feature_scope_source=(
                    "model_artifact_feature_cols" if model_feature_columns else None
                ),
                tokenizer=tokenizer,
                golden_dataset=llm_resources["golden_set"],
                system_prompt=llm_resources["system_prompt"],
                rag_manifest=llm_resources["rag_manifest"],
                guardrail_config=llm_resources["guardrail_config"],
            )

        train_df, eval_df = ResourceLoader.load_traditional_resources(audit_context)
        return ResourceBundle(
            model=model,
            model_metadata=model_metadata,
            model_feature_columns=model_feature_columns,
            model_feature_scope_source=(
                "model_artifact_feature_cols" if model_feature_columns else None
            ),
            training_dataset=train_df,
            evaluation_dataset=eval_df,
            tokenizer=tokenizer,
        )

    @staticmethod
    def load(audit_context) -> ResourceBundle:
        """
        Backward-compatible alias for load_bundle().
        """
        return ResourceLoader.load_bundle(audit_context)
