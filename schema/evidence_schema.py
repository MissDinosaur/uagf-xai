"""Unified evidence result schema used at the pipeline boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


EVIDENCE_STATUSES = frozenset(
    {"completed", "skipped", "failed", "not_applicable"}
)


@dataclass
class EvidenceResult:
    evidence_id: str
    layer: str
    method: str
    status: str
    article_mapping: list[str] = field(default_factory=list)
    summary: str | None = None
    key_findings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    raw_output: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in EVIDENCE_STATUSES:
            raise ValueError(
                f"Unsupported evidence status {self.status!r}. "
                f"Expected one of {sorted(EVIDENCE_STATUSES)}."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evidence_result(
    *,
    evidence_id: str,
    layer: str,
    method: str,
    status: str,
    article_mapping=None,
    summary=None,
    key_findings=None,
    metrics=None,
    artifacts=None,
    limitations=None,
    raw_output=None,
) -> dict[str, Any]:
    return EvidenceResult(
        evidence_id=evidence_id,
        layer=layer,
        method=method,
        status=status,
        article_mapping=list(article_mapping or []),
        summary=summary,
        key_findings=list(key_findings or []),
        metrics=dict(metrics or {}),
        artifacts=list(artifacts or []),
        limitations=list(limitations or []),
        raw_output=dict(raw_output or {}),
    ).to_dict()


def completed_evidence(**kwargs) -> dict[str, Any]:
    return evidence_result(status="completed", **kwargs)


def skipped_evidence(**kwargs) -> dict[str, Any]:
    return evidence_result(status="skipped", **kwargs)


def failed_evidence(**kwargs) -> dict[str, Any]:
    return evidence_result(status="failed", **kwargs)


def not_applicable_evidence(**kwargs) -> dict[str, Any]:
    return evidence_result(status="not_applicable", **kwargs)
