"""Report-facing evidence titles derived from the canonical method catalogue."""

from __future__ import annotations

from schema.method_catalog import METHOD_CATALOG


REPORT_METHOD_TITLES = {
    token: details.display_name for token, details in METHOD_CATALOG.items()
}
REPORT_METHOD_TITLES.update(
    {
        details.result_key: details.display_name
        for details in METHOD_CATALOG.values()
    }
)


def report_method_title(method_token: str, fallback: str | None = None) -> str:
    """Return a polished title without changing the internal method token."""
    return REPORT_METHOD_TITLES.get(method_token, fallback or method_token)
