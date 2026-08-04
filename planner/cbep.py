"""Deterministic Constraint-Based Evidence Planner (CBEP).

CBEP is inspired by constraint satisfaction problem (CSP) theory and
operationalizes evidence selection as a deterministic, constraint-informed
planning procedure. It applies explicit regulatory, governance, system-type,
and task-compatibility constraints to select a minimum sufficient evidence set.

The implementation does not claim to use a generic CSP solver. ``csp_satisfied``
is an upstream S5 audit-status signal that can broaden the evidence sweep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from layers.llm.evidence_methods import (
    LLM_GROUNDING,
    LLM_METHOD_ORDER,
    LLM_PROMPT_FAIRNESS,
    LLM_SELF_CONSISTENCY,
    LLM_SEMANTIC_DRIFT,
    llm_method_display_name,
)
from schema.method_catalog import (
    CATALOG_VERSION,
    LLM_METHOD_TOKENS,
    METHOD_CATALOG,
    METHOD_TASK_COMPATIBILITY,
    TRADITIONAL_METHOD_TOKENS,
    article_method_map,
)

if TYPE_CHECKING:
    from adapters.s4_governance_adapter import GovernanceContext
    from adapters.s5_audit_adapter import AuditContext


_S5_ARTICLE_TRAD = article_method_map("traditional")
_S5_ARTICLE_LLM = article_method_map("llm")

_ALL_TRAD = list(TRADITIONAL_METHOD_TOKENS)
_ALL_LLM = list(LLM_METHOD_TOKENS)
_EXPLAINABILITY_TRAD = ["shap", "lime", "dice"]
_EXPLAINABILITY_LLM = [LLM_GROUNDING]
_DRIFT_TRAD = ["drift"]
_DRIFT_LLM = [LLM_SEMANTIC_DRIFT]

_PRIORITY_THRESHOLD = 2.5
_LOW_GOVERNANCE_THRESHOLD = 2.5
_DOMAIN_KEY_TRANSPARENCY = "transparency"
_DOMAIN_KEY_MONITORING = "monitoring"


def _ordered_union(*lists: list) -> list:
    seen = set()
    result = []
    for values in lists:
        for item in values:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result


def _catalog_order(methods: list[str], is_llm: bool) -> list[str]:
    preferred = _ALL_LLM if is_llm else _ALL_TRAD
    rank = {token: index for index, token in enumerate(preferred)}
    original = {token: index for index, token in enumerate(methods)}
    return sorted(
        methods,
        key=lambda token: (rank.get(token, len(rank)), original[token]),
    )


def _get_domain_score(
    domain_scores: dict,
    keyword: str,
    default: float = 5.0,
) -> float:
    """Return the domain score matching a case-insensitive name fragment."""
    keyword = keyword.lower()
    for name, score in domain_scores.items():
        if keyword in name.lower():
            return float(score)
    return default


def _base_plan(risk_tier: str, is_llm: bool) -> list[str]:
    if is_llm:
        return {
            "minimal": [LLM_GROUNDING],
            "limited": [LLM_GROUNDING, LLM_PROMPT_FAIRNESS],
            "high": list(LLM_METHOD_ORDER),
        }.get(risk_tier, [])
    return {
        "minimal": ["shap"],
        "limited": ["shap", "fairness"],
        "high": ["shap", "fairness", "uncertainty", "drift"],
    }.get(risk_tier, [])


def _normalize_task_type(task_type: str) -> str:
    return str(task_type).strip().lower()


def _apply_task_compatibility_assessment(
    methods: list[str],
    task_type: str,
) -> tuple[list[str], dict]:
    normalized_task = _normalize_task_type(task_type)
    compatible = []
    incompatible = []

    for method in methods:
        allowed_tasks = METHOD_TASK_COMPATIBILITY.get(method)
        if allowed_tasks is None or normalized_task in allowed_tasks:
            compatible.append(method)
        else:
            incompatible.append(method)

    return compatible, {
        "status": "applied",
        "task_type": normalized_task,
        "assessment_policy": "task-method compatibility screening",
        "compatible_methods": compatible,
        "incompatible_methods": incompatible,
        "compatible_count": len(compatible),
        "incompatible_count": len(incompatible),
        "compatibility_map_version": CATALOG_VERSION,
        "note": (
            "Methods incompatible with the current task type were excluded "
            "after CBEP planning."
        ),
    }


def _method_decisions(
    methods: list[str],
    compatibility_trace: dict,
    task_type: str,
    is_llm: bool,
) -> list[dict]:
    incompatible = set(compatibility_trace.get("incompatible_methods", []))
    decisions = []
    for token in (_ALL_LLM if is_llm else _ALL_TRAD):
        if token in methods:
            status = "selected"
            reason = "Required by the merged plan and compatible with the task."
        elif token in incompatible:
            status = "incompatible_filtered"
            reason = (
                f"Excluded because {token} is not compatible with task type "
                f"{task_type}."
            )
        else:
            status = "not_required"
            reason = "Not required by the minimum-sufficient merged plan."
        decisions.append(
            {
                "method": token,
                "display_name": METHOD_CATALOG[token].display_name,
                "status": status,
                "reason": reason,
            }
        )
    return decisions


def plan_evidence(audit, governance=None) -> tuple[list[str], dict]:
    """Select a minimum sufficient method set and return its audit trace."""
    is_llm = str(getattr(audit, "system_type", "")).strip().lower() in {
        "llm",
        "agentic",
    }
    task_type = _normalize_task_type(getattr(audit, "task_type", ""))

    # Step 1: risk-tier base plan.
    base = _base_plan(audit.risk_tier, is_llm)

    # Step 2: EU AI Act article mapping.
    article_map = _S5_ARTICLE_LLM if is_llm else _S5_ARTICLE_TRAD
    article_methods = []
    articles_matched = []
    for article in audit.applicable_articles:
        mapped = article_map.get(article, [])
        if mapped:
            articles_matched.append(article)
            article_methods.extend(mapped)
    methods = _ordered_union(base, article_methods)

    # Step 3: governance-priority adjustments.
    governance_adjustments = []
    if governance is None:
        governance_trace = {"status": "not_provided"}
    else:
        if governance.governance_score < _LOW_GOVERNANCE_THRESHOLD:
            methods = _ordered_union(methods, _ALL_LLM if is_llm else _ALL_TRAD)
            governance_adjustments.append(
                f"governance_score={governance.governance_score:.2f} < "
                f"{_LOW_GOVERNANCE_THRESHOLD} -> full evidence sweep"
            )

        transparency = _get_domain_score(
            governance.domain_scores,
            _DOMAIN_KEY_TRANSPARENCY,
        )
        if transparency < _PRIORITY_THRESHOLD:
            promoted = _EXPLAINABILITY_LLM if is_llm else _EXPLAINABILITY_TRAD
            methods = _ordered_union(promoted, methods)
            governance_adjustments.append(
                f"transparency_score={transparency:.2f} < "
                f"{_PRIORITY_THRESHOLD} -> explainability promoted"
            )

        monitoring = _get_domain_score(
            governance.domain_scores,
            _DOMAIN_KEY_MONITORING,
        )
        if monitoring < _PRIORITY_THRESHOLD:
            promoted = _DRIFT_LLM if is_llm else _DRIFT_TRAD
            methods = _ordered_union(promoted, methods)
            governance_adjustments.append(
                f"monitoring_score={monitoring:.2f} < "
                f"{_PRIORITY_THRESHOLD} -> drift promoted"
            )

        governance_trace = {
            "status": "applied",
            "governance_score": governance.governance_score,
            "governance_verdict": governance.governance_verdict,
            "domain_scores": governance.domain_scores,
            "adjustments": governance_adjustments,
        }

    # Step 4: use the upstream S5 CSP status as an audit signal.
    csp_adjustments = []
    if not audit.csp_satisfied:
        methods = _ordered_union(methods, _ALL_LLM if is_llm else _ALL_TRAD)
        csp_adjustments.append(
            "csp_satisfied=False -> extended evidence sweep enabled"
        )

    # Step 5: task-method compatibility screening.
    if is_llm:
        compatibility_trace = {
            "status": "applied",
            "task_type": task_type,
            "assessment_policy": "LLM pathway compatibility screening",
            "compatible_methods": list(methods),
            "incompatible_methods": [],
            "compatible_count": len(methods),
            "incompatible_count": 0,
            "compatibility_map_version": CATALOG_VERSION,
            "note": "The selected methods belong to the LLM evidence pathway.",
        }
    else:
        methods, compatibility_trace = _apply_task_compatibility_assessment(
            methods,
            task_type,
        )

    # Step 6: stable catalogue order for execution and reporting.
    methods = _catalog_order(methods, is_llm)
    compatibility_trace["compatible_methods"] = list(methods)

    trace = {
        "cbep_version": "3.1",
        "planning_model": "deterministic_constraint_informed",
        "method_catalog_version": CATALOG_VERSION,
        "system_type": audit.system_type,
        "task_type": task_type,
        "modality": audit.modality,
        "risk_tier": audit.risk_tier,
        "application_domain": audit.application_domain,
        "applicable_articles": audit.applicable_articles,
        "articles_matched": articles_matched,
        "blocking_findings_count": len(audit.blocking_findings),
        "csp_satisfied": audit.csp_satisfied,
        "csp_adjustments": csp_adjustments,
        "base_plan": base,
        "article_plan": list(dict.fromkeys(article_methods)),
        "governance_context": governance_trace,
        "task_compatibility_assessment": compatibility_trace,
        "method_decisions": _method_decisions(
            methods,
            compatibility_trace,
            task_type,
            is_llm,
        ),
        "final_plan": methods,
        "final_plan_display_names": [
            llm_method_display_name(method) for method in methods
        ] if is_llm else [METHOD_CATALOG[method].display_name for method in methods],
    }
    return methods, trace


def select_methods_with_trace(
    risk_level: str,
    system_type: str = "traditional_ml",
    s4_json: dict | None = None,
    s5_json: dict | None = None,
) -> tuple[list[str], dict]:
    """Backward-compatible dictionary-input wrapper around ``plan_evidence``."""
    from adapters.s4_governance_adapter import GovernanceAdapter
    from adapters.s5_audit_adapter import AuditAdapter, AuditContext

    if s5_json:
        audit = AuditAdapter.from_audit_report(s5_json)
    else:
        audit = AuditContext(
            system_type=system_type,
            modality="tabular",
            application_domain="unknown",
            risk_tier=risk_level,
            applicable_articles=[],
        )
    governance = GovernanceAdapter.from_cgsa_report(s4_json) if s4_json else None
    return plan_evidence(audit, governance)


def select_methods(
    risk_level: str,
    system_type: str = "traditional_ml",
    s4_json: dict | None = None,
    s5_json: dict | None = None,
) -> list[str]:
    """Return only the planned method tokens for legacy callers."""
    methods, _ = select_methods_with_trace(
        risk_level,
        system_type,
        s4_json,
        s5_json,
    )
    return methods
