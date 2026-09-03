from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

import uagf_xai
from adapters.s5_audit_adapter import AuditContext
from report import report_generator
from schema.evidence_schema import completed_evidence
from uagf_xai import cli


def test_public_package_import_and_version():
    assert uagf_xai.__version__ == "1.0.0"


def test_public_api_contains_only_supported_entry_points():
    expected = {
        "audit",
        "AuditAdapter",
        "AuditContext",
        "GovernanceAdapter",
        "GovernanceContext",
    }
    assert set(uagf_xai.__all__) == expected
    assert callable(uagf_xai.audit)


def test_report_template_resolves_outside_repository_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    context = AuditContext(
        system_type="traditional_ml",
        modality="tabular",
        risk_tier="high",
        task_type="binary_classification",
        provider_name="Packaging Test",
    )
    result = completed_evidence(
        evidence_id="EXP-SHAP",
        layer="explainability",
        method="SHAP",
        summary="Packaging test evidence.",
    )
    results = {
        "explainability": result,
        "_cbep_trace": {
            "type": "cbep_planning_trace",
            "method": "CBEP",
            "base_plan": ["shap"],
            "article_plan": [],
            "final_plan": ["shap"],
            "task_compatibility_assessment": {
                "incompatible_methods": [],
                "incompatibility_reasons": {},
            },
        },
    }

    output = report_generator.generate_report(
        results,
        "high",
        provider_name="Packaging Test",
        audit_context=context,
        generate_pdf=False,
    )

    output_path = Path(output).resolve()
    assert output_path.is_file()
    assert output_path.is_relative_to(tmp_path.resolve())
    assert "UAGF-XAI Audit Report" in output_path.read_text(encoding="utf-8")


def test_report_template_is_present_beside_report_package():
    template = Path(report_generator.__file__).resolve().parent / "templates"
    assert (template / "report_template.html").is_file()


def test_cli_module_imports_and_help_exits_successfully(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["uagf-xai", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert "Run UAGF-XAI audit workflow" in capsys.readouterr().out


def test_public_metadata_contains_no_personal_identity():
    project_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    license_text = (project_root / "LICENSE").read_text(encoding="utf-8")

    assert "authors" not in metadata
    assert "maintainers" not in metadata
    assert "urls" not in metadata
    assert "UAGF-XAI contributors" in license_text
