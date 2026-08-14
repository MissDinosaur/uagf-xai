from __future__ import annotations

import json
import re
from types import SimpleNamespace

from report.method_titles import report_method_title
from report import runtime_metadata
from report.runtime_metadata import (
    collect_dependency_versions,
    collect_git_metadata,
    collect_input_provenance,
    collect_reproducibility_metadata,
    collect_runtime_environment,
    hash_resource,
    infer_record_count,
    public_report_reproducibility_metadata,
    sha256_directory_manifest,
    sha256_file,
)


def test_runtime_environment_contains_python_and_platform():
    metadata = collect_runtime_environment("2026-07-27T12:00:00+02:00")

    assert metadata["run_timestamp"] == "2026-07-27T12:00:00+02:00"
    assert metadata["python_version"]
    assert metadata["platform"]
    assert "package_versions" not in metadata


def test_runtime_environment_formats_timestamp_and_python_concisely():
    metadata = collect_runtime_environment("2026-07-27T12:00:00.987654+02:00")

    assert metadata["run_timestamp"] == "2026-07-27T12:00:00+02:00"
    assert re.fullmatch(r"\d+\.\d+\.\d+ \[[^\]]+\]", metadata["python_version"])


def test_record_count_supports_frames_lists_and_nested_golden_sets():
    class FrameLike:
        shape = (17, 4)

    assert infer_record_count(FrameLike()) == 17
    assert infer_record_count([1, 2, 3]) == 3
    assert infer_record_count({"metadata": {}, "test_cases": [1, 2, 3, 4]}) == 4
    assert infer_record_count({"metadata": {}}) is None


def test_report_titles_are_polished_without_changing_tokens():
    assert report_method_title("llm_grounding") == "LLM-E1 — Grounding Score"
    assert (
        report_method_title("shap")
        == "Explainability Evidence — SHAP Feature Attribution"
    )
    assert (
        report_method_title("dice")
        == "Explainability Evidence — DiCE Counterfactual Explanation"
    )
    assert report_method_title("llm_grounding") != "llm_grounding"


def test_core_versions_are_collected_and_missing_optional_packages_are_explicit(
    monkeypatch,
):
    def fake_version(name):
        if name in {"numpy", "scikit-learn"}:
            return {"numpy": "2.1.0", "scikit-learn": "1.8.0"}[name]
        raise runtime_metadata.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runtime_metadata.metadata, "version", fake_version)
    versions = collect_dependency_versions()

    assert versions["numpy"] == "2.1.0"
    assert versions["scikit-learn"] == "1.8.0"
    assert versions["transformers"] == "not_installed"
    assert set(runtime_metadata.DEPENDENCY_DISTRIBUTIONS) <= set(versions)


def test_file_hash_is_deterministic_and_changes_with_content(tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b'{"value": 1}')
    first = sha256_file(path)

    assert first == sha256_file(path)
    path.write_bytes(b'{"value": 2}')
    assert sha256_file(path) != first


def test_directory_manifest_hash_is_order_independent_and_content_sensitive(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "b.txt").write_text("B", encoding="utf-8")
    (first / "a.txt").write_text("A", encoding="utf-8")
    (second / "a.txt").write_text("A", encoding="utf-8")
    (second / "b.txt").write_text("B", encoding="utf-8")

    first_hash = sha256_directory_manifest(first)
    second_hash = sha256_directory_manifest(second)
    assert first_hash == second_hash
    assert first_hash["included_file_count"] == 2

    (second / "b.txt").write_text("changed", encoding="utf-8")
    assert sha256_directory_manifest(second)["sha256"] != first_hash["sha256"]


def test_git_unavailable_is_structured(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runtime_metadata.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    result = collect_git_metadata(tmp_path)

    assert result == {
        "status": "unavailable",
        "commit_sha": "unavailable",
        "branch": "unavailable",
        "dirty": "unavailable",
    }


def test_input_provenance_hashes_s4_s5_model_and_datasets(tmp_path):
    paths = {}
    for name in ("s4", "s5", "model", "train", "evaluation"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("utf-8"))
        paths[name] = path
    audit_context = SimpleNamespace(
        system_type="traditional_ml",
        source_json_path=paths["s5"],
        model_artifact_uri=str(paths["model"]),
        training_dataset_uri=str(paths["train"]),
        evaluation_dataset_uri=str(paths["evaluation"]),
    )
    governance_context = SimpleNamespace(source_json_path=paths["s4"])

    provenance = collect_input_provenance(
        audit_context, governance_context, project_root=tmp_path
    )

    for label in (
        "s4_json",
        "s5_json",
        "model_artifact",
        "training_dataset",
        "evaluation_dataset",
    ):
        assert provenance[label]["status"] == "available"
        assert len(provenance[label]["sha256"]) == 64
    assert provenance["golden_set"]["status"] == "not_applicable"


def test_legalmind_directory_and_golden_set_hashes_are_available():
    s5_path = "data/04_legalmindd_ai_ltd/s5_legalmindd_ai_ltd_audit_state.json"
    with open(s5_path, encoding="utf-8") as input_file:
        data = json.load(input_file)
    from adapters.s5_audit_adapter import AuditAdapter

    audit_context = AuditAdapter.from_audit_report(data)
    audit_context.source_json_path = s5_path
    provenance = collect_input_provenance(audit_context, None)

    assert provenance["model_artifact"]["kind"] == "directory_manifest"
    assert provenance["model_artifact"]["included_file_count"] > 0
    assert provenance["golden_set"]["status"] == "available"
    assert len(provenance["golden_set"]["sha256"]) == 64


def test_local_provenance_uses_local_context_label_not_s5(tmp_path):
    local_config = tmp_path / "s6_local_config.json"
    local_config.write_text("{}", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    embedding = tmp_path / "embedding"
    embedding.mkdir()
    drift = tmp_path / "drift.json"
    drift.write_text("{}", encoding="utf-8")
    fairness = tmp_path / "fairness.json"
    fairness.write_text("{}", encoding="utf-8")
    audit_context = SimpleNamespace(
        system_type="llm",
        model_artifact_uri=str(model),
        golden_set_uri=None,
        system_prompt_uri=None,
        rag_manifest_uri=None,
        guardrail_config_uri=None,
    )
    evidence_config = SimpleNamespace(
        evaluation_embedding_model_uri=str(embedding),
        semantic_drift_dataset_uri=str(drift),
        fairness_prompt_pairs_uri=str(fairness),
    )

    provenance = collect_input_provenance(
        audit_context,
        None,
        project_root=tmp_path,
        runtime_context={
            "runtime_mode": "local",
            "local_validation_config_path": str(local_config),
            "llm_evidence_config": evidence_config,
        },
    )

    assert "local_validation_context" in provenance
    assert provenance["local_validation_context"]["status"] == "available"
    assert "s5_json" not in provenance
    assert provenance["evaluation_embedding_model"]["status"] == "available"


def test_reproducibility_record_is_stable_when_dynamic_fields_are_fixed(tmp_path):
    audit_context = SimpleNamespace(system_type="traditional_ml")
    fixed = {
        "run_timestamp": "2026-08-06T12:30:00+02:00",
        "run_id": "fixed-run-id",
        "total_runtime_seconds": 1.25,
        "cbep_version": "3.1",
    }

    first = collect_reproducibility_metadata(
        audit_context, None, fixed, project_root=tmp_path
    )
    second = collect_reproducibility_metadata(
        audit_context, None, fixed, project_root=tmp_path
    )

    assert first["reproducibility"] == second["reproducibility"]


def test_public_reproducibility_omits_git_but_internal_metadata_keeps_it(
    monkeypatch, tmp_path
):
    git_state = {
        "status": "available",
        "commit_sha": "a" * 40,
        "branch": "private-research-branch",
        "dirty": True,
    }
    monkeypatch.setattr(runtime_metadata, "collect_git_metadata", lambda root: git_state)

    internal = collect_reproducibility_metadata(project_root=tmp_path)
    public = public_report_reproducibility_metadata(internal)

    assert internal["internal_reproducibility_metadata"]["project_state"]["git"] == git_state
    assert internal["reproducibility"]["project_state"]["git"] == git_state
    assert "git" not in public["project_state"]
    assert "git" not in public["reproducibility"]["project_state"]
    assert public["project_state"]["evidence_schema_version"] == "1.0"
    assert public["project_state"]["report_version"] == "1.0"


def test_hash_resource_does_not_expose_external_absolute_path(tmp_path):
    path = tmp_path / "secret-location-name.json"
    path.write_text("{}", encoding="utf-8")

    record = hash_resource(path, project_root=tmp_path / "different-root")

    assert record["path"] == path.name
    assert str(tmp_path) not in record["path"]
