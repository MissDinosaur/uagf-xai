"""
GovernanceAdapter  —  UAGF-XAI S6 / Intake Layer
==================================================
Reads a CGSA canonical report (S4 / UAGF-GMM output) and projects it into
the GovernanceContext data-transfer object consumed by the CBEP planner.

Expected S4 / CGSA canonical report structure
---------------------------------------------
{
    "overall_scores": {
        "composite_maturity_score": 2.58,
        "governance_verdict":       "FAIL"
    },
    "domains": [
        {"domain_id": "D1", "domain_name": "Risk Management",                    "domain_score": 2.86},
        {"domain_id": "D2", "domain_name": "Data Governance",                    "domain_score": 2.86},
        {"domain_id": "D3", "domain_name": "Model Development and Testing",      "domain_score": 2.71},
        {"domain_id": "D4", "domain_name": "Transparency and Explainability",    "domain_score": 2.00},
        {"domain_id": "D5", "domain_name": "Human Oversight and Accountability", "domain_score": 2.50},
        {"domain_id": "D6", "domain_name": "Monitoring and Incident Response",   "domain_score": 2.10}
    ]
}

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Data-Transfer Object
# ---------------------------------------------------------------------------

@dataclass
class GovernanceContext:
    """
    Governance information provided by S4 (UAGF-GMM / CGSA).

    All scores use the CGSA 0-5 maturity scale::

        1 = Initial       2 = Developing    3 = Defined
        4 = Managed       5 = Optimised

    ``domain_scores`` uses the canonical domain name string as key
    (e.g. "Transparency and Explainability") so that new domains added
    by S4 are automatically forwarded to CBEP without any code change.
    """

    governance_score:   float           # composite score, 0-5
    governance_verdict: str             # COMPLIANT | COMPLIANT_WITH_OBSERVATIONS | CONDITIONAL_PASS | FAIL
    domain_scores: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class GovernanceAdapter:
    """
    Translates a CGSA canonical report dict into a GovernanceContext DTO.

    Usage::

        import json
        from adapters.governance_adapter import GovernanceAdapter

        with open("s4_cgsa_report.json") as f:
            data = json.load(f)
        ctx = GovernanceAdapter.from_cgsa_report(data)
    """

    @classmethod
    def from_cgsa_report(cls, data: dict) -> GovernanceContext:
        """
        Parse a CGSA canonical report dict and return GovernanceContext.

        Structural tolerances:
        - Domain scores accept ``domain_score`` first, then legacy ``avg_score``
          and ``score`` field names
        - ``overall_scores.composite_maturity_score`` is the primary global
          score field, with legacy fallbacks preserved
        - New domains are forwarded automatically via the dict; no hardcoding
        """
        overall_scores = data.get("overall_scores", {})
        composite_score = float(overall_scores.get(
                    "composite_maturity_score",
                    data.get("governance_score", 0.0),
                )
        )
        verdict = str(overall_scores.get("governance_verdict", "FAIL"))

        # Collect per-domain scores keyed by the canonical domain_name string
        domain_scores: dict[str, float] = {}
        for dom in data.get("domains", []):
            name = str(dom.get("domain_name", "")).strip()
            score = float(dom.get("domain_score", 0.0))
            if name:
                domain_scores[name] = score

        return GovernanceContext(
            governance_score=composite_score,
            governance_verdict=verdict,
            domain_scores=domain_scores,
        )

    @classmethod
    def fallback(cls) -> GovernanceContext:
        """
        Return a safe neutral context when no S4 data is available.

        ``domain_scores`` is empty; CBEP uses 5.0 as the default for any
        missing domain, so no priority-promotion or extended sweeps are
        triggered by the absence of governance data.
        """
        return GovernanceContext(
            governance_score=5.0,
            governance_verdict="PASS",
            domain_scores={},
        )
