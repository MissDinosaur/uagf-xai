"""Collect structured runtime and provenance metadata for audit reports."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
from pathlib import Path
import platform as platform_module
import subprocess
from typing import Any
from uuid import uuid4

from resources.artifact_utils import resolve_artifact_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_SCHEMA_VERSION = "1.0"
REPORT_VERSION = "1.0"

DEPENDENCY_DISTRIBUTIONS = {
    "scikit-learn": ("scikit-learn",),
    "numpy": ("numpy",),
    "scipy": ("scipy",),
    "pandas": ("pandas",),
    "joblib": ("joblib",),
    "shap": ("shap",),
    "lime": ("lime",),
    "dice-ml": ("dice-ml",),
    "fairlearn": ("fairlearn",),
    "mapie": ("MAPIE", "mapie"),
    "evidently": ("evidently",),
    "jinja2": ("Jinja2", "jinja2"),
    "reportlab": ("reportlab",),
    "torch": ("torch",),
    "transformers": ("transformers",),
    "sentence-transformers": ("sentence-transformers",),
    "peft": ("peft",),
}

LLM_DEPENDENCIES = frozenset(
    {"torch", "transformers", "sentence-transformers", "peft"}
)

_EXCLUDED_DIRECTORY_PARTS = frozenset(
    {".git", ".pytest_cache", "__pycache__", "cache", "outputs", "tmp", "temp"}
)


def collect_runtime_environment(run_timestamp: str | None = None) -> dict[str, Any]:
    """Return stable, report-ready runtime environment metadata."""
    if run_timestamp:
        try:
            timestamp_value = datetime.fromisoformat(
                run_timestamp.replace("Z", "+00:00")
            ).replace(microsecond=0).isoformat()
        except ValueError:
            timestamp_value = run_timestamp
    else:
        timestamp_value = (
            datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()
        )
    python_version = platform_module.python_version()
    compiler = platform_module.python_compiler()
    if compiler:
        python_version = f"{python_version} [{compiler}]"
    return {
        "run_timestamp": timestamp_value,
        "python_version": python_version,
        "platform": platform_module.platform(),
    }


def collect_dependency_versions() -> dict[str, str]:
    """Resolve package versions without importing optional or heavyweight packages."""
    versions = {}
    for display_name, distribution_names in DEPENDENCY_DISTRIBUTIONS.items():
        versions[display_name] = "not_installed"
        for distribution_name in distribution_names:
            try:
                versions[display_name] = metadata.version(distribution_name)
                break
            except metadata.PackageNotFoundError:
                continue
    return versions


def sha256_file(path: str | Path) -> str:
    """Calculate SHA-256 over the exact bytes of one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _include_manifest_file(path: Path, base_path: Path) -> bool:
    relative = path.relative_to(base_path)
    lowered_parts = {part.lower() for part in relative.parts}
    if lowered_parts & _EXCLUDED_DIRECTORY_PARTS:
        return False
    lowered_name = path.name.lower()
    return not (
        lowered_name.startswith("~")
        or lowered_name.endswith((".tmp", ".temp", ".pyc"))
    )


def sha256_directory_manifest(path: str | Path) -> dict[str, Any]:
    """Hash a deterministic, path-sorted manifest of regular directory files."""
    base_path = Path(path)
    files = sorted(
        (
            item.relative_to(base_path).as_posix(),
            item,
        )
        for item in base_path.rglob("*")
        if item.is_file() and _include_manifest_file(item, base_path)
    )
    digest = hashlib.sha256()
    for relative_path, file_path in files:
        file_hash = sha256_file(file_path)
        digest.update(f"{relative_path}\0{file_hash}\n".encode("utf-8"))
    return {"sha256": digest.hexdigest(), "included_file_count": len(files)}


def _public_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def hash_resource(path_or_uri, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return a non-secret, public-path hash record for a file or directory."""
    if not path_or_uri:
        return {
            "status": "not_provided",
            "path": "not_provided",
            "sha256": "not_provided",
            "kind": "not_provided",
            "included_file_count": "not_applicable",
        }
    project_root = Path(project_root)
    path = resolve_artifact_path(str(path_or_uri))
    public_path = _public_path(path, project_root)
    if not path.exists():
        return {
            "status": "unavailable",
            "path": public_path,
            "sha256": "unavailable",
            "kind": "unavailable",
            "included_file_count": "not_applicable",
        }
    try:
        if path.is_dir():
            manifest = sha256_directory_manifest(path)
            return {
                "status": "available",
                "path": public_path,
                "sha256": manifest["sha256"],
                "kind": "directory_manifest",
                "included_file_count": manifest["included_file_count"],
            }
        return {
            "status": "available",
            "path": public_path,
            "sha256": sha256_file(path),
            "kind": "file",
            "included_file_count": "not_applicable",
        }
    except OSError:
        return {
            "status": "unavailable",
            "path": public_path,
            "sha256": "unavailable",
            "kind": "unavailable",
            "included_file_count": "not_applicable",
        }


def collect_git_metadata(project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Collect Git state without failing when Git or repository metadata is absent."""
    project_root = Path(project_root)

    def git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(git("status", "--porcelain"))
        return {
            "status": "available",
            "commit_sha": commit,
            "branch": branch,
            "dirty": dirty,
        }
    except (OSError, subprocess.SubprocessError):
        return {
            "status": "unavailable",
            "commit_sha": "unavailable",
            "branch": "unavailable",
            "dirty": "unavailable",
        }


def _not_applicable_resource() -> dict[str, Any]:
    return {
        "status": "not_applicable",
        "path": "not_applicable",
        "sha256": "not_applicable",
        "kind": "not_applicable",
        "included_file_count": "not_applicable",
    }


def collect_input_provenance(
    audit_context,
    governance_context,
    project_root: str | Path = PROJECT_ROOT,
    runtime_context=None,
) -> dict[str, dict[str, Any]]:
    """Hash S4/S5 inputs and every resource referenced by the active contract."""
    runtime_context = runtime_context or {}
    runtime_mode = runtime_context.get("runtime_mode", "s5")
    evidence_config = runtime_context.get("llm_evidence_config")
    is_llm = str(getattr(audit_context, "system_type", "") or "").lower() in {
        "llm",
        "agentic",
    }
    records = {
        "s4_json": hash_resource(
            getattr(governance_context, "source_json_path", None), project_root
        ),
        "model_artifact": hash_resource(
            getattr(audit_context, "model_artifact_uri", None), project_root
        ),
    }
    if runtime_mode == "local":
        records["local_validation_context"] = hash_resource(
            runtime_context.get("local_validation_config_path"), project_root
        )
    else:
        records["s5_json"] = hash_resource(
            getattr(audit_context, "source_json_path", None), project_root
        )
    traditional_fields = {
        "training_dataset": "training_dataset_uri",
        "evaluation_dataset": "evaluation_dataset_uri",
    }
    llm_fields = {
        "golden_set": "golden_set_uri",
        "system_prompt": "system_prompt_uri",
        "rag_manifest": "rag_manifest_uri",
        "guardrail_config": "guardrail_config_uri",
    }
    for label, field_name in traditional_fields.items():
        records[label] = (
            _not_applicable_resource()
            if is_llm
            else hash_resource(getattr(audit_context, field_name, None), project_root)
        )
    for label, field_name in llm_fields.items():
        records[label] = (
            hash_resource(getattr(audit_context, field_name, None), project_root)
            if is_llm
            else _not_applicable_resource()
        )
    s6_llm_fields = {
        "evaluation_embedding_model": "evaluation_embedding_model_uri",
        "semantic_drift_validation": "semantic_drift_dataset_uri",
        "fairness_prompt_pairs": "fairness_prompt_pairs_uri",
    }
    for label, field_name in s6_llm_fields.items():
        value = getattr(evidence_config, field_name, None)
        records[label] = (
            hash_resource(value, project_root)
            if is_llm and evidence_config is not None
            else _not_applicable_resource()
        )
    return records


def collect_reproducibility_metadata(
    audit_context=None,
    governance_context=None,
    runtime_context=None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Collect one complete reproducibility record for an audit report."""
    runtime_context = dict(runtime_context or {})
    environment = collect_runtime_environment(runtime_context.get("run_timestamp"))
    runtime = {
        **environment,
        "run_id": runtime_context.get("run_id") or str(uuid4()),
        "total_runtime_seconds": runtime_context.get(
            "total_runtime_seconds", "unavailable"
        ),
    }
    project_state = {
        "git": collect_git_metadata(project_root),
        "cbep_version": runtime_context.get("cbep_version", "unavailable"),
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "report_version": REPORT_VERSION,
    }
    reproducibility = {
        "runtime": runtime,
        "dependencies": collect_dependency_versions(),
        "project_state": project_state,
        "input_provenance": collect_input_provenance(
            audit_context,
            governance_context,
            project_root,
            runtime_context,
        ),
    }
    return {
        **environment,
        "run_id": runtime["run_id"],
        "dependency_versions": reproducibility["dependencies"],
        "project_state": project_state,
        "input_provenance": reproducibility["input_provenance"],
        "internal_reproducibility_metadata": reproducibility,
        "reproducibility": reproducibility,
    }


def public_report_reproducibility_metadata(internal_metadata: dict) -> dict:
    """Project internal metadata into a minimal user-facing report context."""
    internal_reproducibility = internal_metadata.get("reproducibility") or {}
    public_reproducibility = {
        "runtime": dict(internal_reproducibility.get("runtime") or {}),
        "dependencies": dict(internal_reproducibility.get("dependencies") or {}),
        "input_provenance": dict(
            internal_reproducibility.get("input_provenance") or {}
        ),
    }
    return {
        "run_timestamp": internal_metadata.get("run_timestamp"),
        "python_version": internal_metadata.get("python_version"),
        "platform": internal_metadata.get("platform"),
        "run_id": internal_metadata.get("run_id"),
        "dependency_versions": public_reproducibility["dependencies"],
        "input_provenance": public_reproducibility["input_provenance"],
        "reproducibility": public_reproducibility,
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
        for key in ("golden_set", "records", "items", "data", "examples"):
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
