"""Collect runtime metadata for reproducible audit reports."""

from __future__ import annotations

from datetime import datetime, timezone
import platform as platform_module
import sys
from typing import Any


def collect_runtime_environment(run_timestamp: str | None = None) -> dict[str, Any]:
    """Return stable, report-ready runtime environment metadata."""
    return {
        "run_timestamp": run_timestamp or datetime.now(timezone.utc).astimezone().isoformat(),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform_module.platform(),
    }


def infer_record_count(value: Any) -> int | None:
    """Infer a dataset or golden-set record count without mutating the input."""
    if value is None or isinstance(value, (str, bytes)):
        return None

    if isinstance(value, (list, tuple)):
        return len(value)

    if hasattr(value, "shape"):
        try:
            return int(value.shape[0])
        except (IndexError, TypeError, ValueError):
            pass

    if isinstance(value, dict):
        preferred_keys = ("golden_set", "records", "items", "data", "examples")
        for key in preferred_keys:
            candidate = value.get(key)
            if isinstance(candidate, (list, tuple)):
                return len(candidate)

        candidate_lengths = [
            len(candidate)
            for candidate in value.values()
            if isinstance(candidate, (list, tuple))
        ]
        return max(candidate_lengths) if candidate_lengths else None

    try:
        return len(value)
    except (TypeError, ValueError):
        return None
