"""
Unified resource loader.
"""

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
    def load_datasets(audit_context):
        return DatasetLoader.load(audit_context)

    @staticmethod
    def load_bundle(audit_context) -> ResourceBundle:
        model = ResourceLoader.load_model(audit_context)
        tokenizer = ResourceLoader.load_tokenizer(audit_context)
        train_df, eval_df = ResourceLoader.load_datasets(audit_context)
        return ResourceBundle(
            model=model,
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
