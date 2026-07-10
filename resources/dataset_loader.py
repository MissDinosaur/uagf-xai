"""
Dataset loading utilities.
"""

import pandas as pd

from .artifact_utils import resolve_artifact_path


class DatasetLoader:

    @staticmethod
    def load_training(audit_context):
        uri = audit_context.training_dataset_uri
        local = resolve_artifact_path(uri)
        return pd.read_csv(local)

    @staticmethod
    def load_evaluation(audit_context):
        uri = audit_context.evaluation_dataset_uri
        local = resolve_artifact_path(uri)
        return pd.read_csv(local)

    @staticmethod
    def load(audit_context):
        train = DatasetLoader.load_training(audit_context)
        test = DatasetLoader.load_evaluation(audit_context)
        return train, test
