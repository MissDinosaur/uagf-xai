import pytest

from schema.evidence_schema import (
    EVIDENCE_STATUSES,
    EvidenceResult,
    completed_evidence,
    failed_evidence,
    not_applicable_evidence,
    skipped_evidence,
)


def test_evidence_result_defaults_and_serialization():
    result = EvidenceResult("E1", "test", "Method", "completed")

    assert result.to_dict() == {
        "evidence_id": "E1",
        "layer": "test",
        "method": "Method",
        "status": "completed",
        "article_mapping": [],
        "summary": None,
        "key_findings": [],
        "metrics": {},
        "artifacts": [],
        "limitations": [],
        "raw_output": {},
    }


@pytest.mark.parametrize(
    ("factory", "expected_status"),
    [
        (completed_evidence, "completed"),
        (skipped_evidence, "skipped"),
        (failed_evidence, "failed"),
        (not_applicable_evidence, "not_applicable"),
    ],
)
def test_evidence_factories_set_allowed_status(factory, expected_status):
    result = factory(evidence_id="E1", layer="test", method="Method")

    assert result["status"] == expected_status
    assert result["status"] in EVIDENCE_STATUSES


def test_invalid_evidence_status_is_rejected():
    with pytest.raises(ValueError, match="Unsupported evidence status"):
        EvidenceResult("E1", "test", "Method", "unknown")


def test_raw_output_and_articles_are_preserved():
    result = completed_evidence(
        evidence_id="E1",
        layer="test",
        method="Method",
        article_mapping=["Art. 13"],
        raw_output={"nested": {"value": 7}},
    )

    assert result["article_mapping"] == ["Art. 13"]
    assert result["raw_output"] == {"nested": {"value": 7}}
