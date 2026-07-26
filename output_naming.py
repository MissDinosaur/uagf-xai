from __future__ import annotations

import re
from pathlib import Path


def _sanitize_namespace(value: str, fallback: str = "audit") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _sanitize_provider_name(value: str, fallback: str = "audit") -> str:
    cleaned = str(value).strip()
    cleaned = re.sub(r"[<>:\"/\\|?*]", "_", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_").lower()
    return cleaned or fallback


def derive_output_namespace(s5_json_path: str | None, fallback: str = "audit") -> str:
    if not s5_json_path:
        return fallback

    path = Path(s5_json_path)
    parent_name = path.parent.name.strip()

    if (
        parent_name
        and parent_name.lower() not in {"data", "outputs"}
        and not parent_name.lower().startswith("z_")
    ):
        return _sanitize_namespace(parent_name, fallback=fallback)

    stem = path.stem.strip()
    stem = re.sub(r"(?i)^s5[_-]*", "", stem)
    stem = re.sub(r"(?i)(_audit_context|_audit_state|_context|_state)$", "", stem)
    stem = stem or fallback
    return _sanitize_namespace(stem, fallback=fallback)


def resolve_output_prefix(
    provider_name: str | None,
    fallback: str = "audit",
    s5_json_path: str | None = None,
) -> str:
    if provider_name:
        return _sanitize_provider_name(provider_name, fallback=fallback)
    if s5_json_path:
        return derive_output_namespace(s5_json_path, fallback=fallback)
    return fallback


def build_output_path(
    output_dir: str,
    provider_name: str | None,
    suffix: str,
    extension: str,
    fallback: str = "audit",
    s5_json_path: str | None = None,
) -> str:
    safe_namespace = resolve_output_prefix(
        provider_name,
        fallback=fallback,
        s5_json_path=s5_json_path,
    )
    safe_suffix = suffix.lstrip("_")
    safe_extension = extension if extension.startswith(".") else f".{extension}"
    filename = f"{safe_namespace}_{safe_suffix}{safe_extension}"
    return str(Path("outputs") / output_dir / filename)

