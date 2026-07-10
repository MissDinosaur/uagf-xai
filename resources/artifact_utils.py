"""
Shared helpers for resolving S5 artifact URIs and normalizing loaded objects.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

def resolve_artifact_path(uri: str, cache_root: str = "data/cache") -> Path:
    """
    Convert a storage URI into a local filesystem path.

    Supported schemes:
    - file:// -> direct local path
    - minio:// -> local cache filename
    - plain path -> returned unchanged
    """
    if uri.startswith("file://"):
        return Path(uri.replace("file://", "", 1))

    if uri.startswith("minio://"):
        filename = uri.split("/")[-1]
        return Path(cache_root) / filename

    return Path(uri)


def unwrap_artifact(value):
    """
    Extract the most useful object from common artifact container shapes.

    This keeps callers agnostic to whether a serialized object stores the
    estimator directly or wraps it together with metadata.
    """
    if _is_model_like(value):
        return value

    if isinstance(value, dict):
        for key in ("model", "estimator", "pipeline"):
            candidate = value.get(key)
            if candidate is not None:
                return unwrap_artifact(candidate)

        for candidate in value.values():
            unwrapped = unwrap_artifact(candidate)
            if _is_model_like(unwrapped):
                return unwrapped

    if isinstance(value, (list, tuple)):
        for candidate in value:
            unwrapped = unwrap_artifact(candidate)
            if _is_model_like(unwrapped):
                return unwrapped

    return value


def _is_model_like(obj) -> bool:
    return any(hasattr(obj, attr) for attr in ("predict", "predict_proba", "fit"))


def to_numeric_frame(frame):
    """
    Return a copy of a tabular frame with categorical/object columns encoded.

    This is mainly used by explainability and uncertainty layers that expect
    float-compatible numeric inputs.
    """
    if not hasattr(frame, "columns"):
        return frame

    out = frame.copy()
    for col in out.columns:
        series = out[col]
        if hasattr(series, "cat"):
            out[col] = series.cat.codes.astype("float64")
        elif getattr(series, "dtype", None) == object:
            out[col] = series.astype("category").cat.codes.astype("float64")
    return out
