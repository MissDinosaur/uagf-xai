"""
Constraint-Based Evidence Planner (CBEP)  —  UAGF-XAI / S6  v3.0
==================================================================
Selects the *minimum sufficient* set of evidence methods by combining
two typed inputs from upstream UAGF platform stages:

  GovernanceContext (S4 / UAGF-GMM / CGSA)
      Overall governance maturity score and per-domain scores (0-5 scale).
      Low scores drive priority-promotion of related evidence methods.

  AuditContext (S5 / UAGF-TAM / AAA)
      EU AI Act applicable articles, risk tier, system modality, and CSP
      satisfiability.  Articles directly select evidence methods.

Planning algorithm
------------------
1. Base plan     -- minimum method set for the declared risk tier.
2. Article plan  -- method set derived from applicable EU AI Act articles.
3. Merge         -- ordered union of base + article methods.
4. Gov. priority -- transparency_score < 2.5  -> explainability promoted to
                    front of queue; monitoring_score < 2.5 -> drift promoted.
5. CSP sweep     -- if audit.csp_satisfied == False, add full method
                    sweep (all available methods for the modality branch).

EU AI Act article -> evidence method mapping
--------------------------------------------
  Art. 9   risk management system        -> uncertainty (MAPIE)
  Art. 10  data governance / fairness    -> fairness (Fairlearn)
  Art. 13  transparency / explainability -> shap, lime, dice
  Art. 14  human oversight               -> uncertainty
  Art. 15  robustness & accuracy         -> uncertainty, drift
  Art. 61  post-market monitoring        -> drift (Evidently)

LLM / agentic equivalents use the llm_ prefixed method tokens.

Public API
----------
  plan_evidence(audit, governance)             -> (methods, trace)   [primary]
  select_methods_with_trace(risk_level, ...)   -> (methods, trace)   [compat]
  select_methods(risk_level, ...)              -> methods             [compat]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adapters.s4_governance_adapter import GovernanceContext
    from adapters.s5_audit_adapter import AuditContext


# ---------------------------------------------------------------------------
# Lookup tables: article -> method tokens
# ---------------------------------------------------------------------------

_S5_ARTICLE_TRAD: dict[str, list[str]] = {
    "Art9":  ["uncertainty"],
    "Art10": ["fairness"],
    "Art13": ["shap", "lime", "dice"],
    "Art14": ["uncertainty"],
    "Art15": ["uncertainty", "drift"],
    "Art61": ["drift"],
}

_S5_ARTICLE_LLM: dict[str, list[str]] = {
    "Art9":  ["llm_uncertainty"],
    "Art10": ["llm_fairness"],
    "Art13": ["llm_explainability"],
    "Art14": ["llm_uncertainty"],
    "Art15": ["llm_uncertainty", "llm_drift"],
    "Art61": ["llm_drift"],
}

_ALL_TRAD = ["shap", "lime", "fairness", "uncertainty", "drift", "dice"]
_ALL_LLM  = ["llm_explainability", "llm_fairness", "llm_uncertainty", "llm_drift"]

_EXPLAINABILITY_TRAD = ["shap", "lime", "dice"]
_EXPLAINABILITY_LLM  = ["llm_explainability"]
_DRIFT_TRAD          = ["drift"]
_DRIFT_LLM           = ["llm_drift"]

METHOD_TASK_COMPATIBILITY: dict[str, set[str]] = {
    "shap": {
        "binary_classification",
        "multiclass_classification",
        "regression",
        "forecasting",
        "anomaly_detection",
    },
    "lime": {
        "binary_classification",
        "multiclass_classification",
        "regression",
    },
    "dice": {
        "binary_classification",
        "multiclass_classification",
        "regression",
    },
    "fairness": {
        "binary_classification",
        "multiclass_classification",
    },
    "uncertainty": {
        "binary_classification",
        "multiclass_classification",
        "regression",
    },
    "drift": {
        "binary_classification",
        "multiclass_classification",
        "regression",
        "forecasting",
        "anomaly_detection",
    },
}

_PRIORITY_THRESHOLD       = 2.5
_LOW_GOVERNANCE_THRESHOLD = 2.5

# Domain name keywords used for governance-driven priority promotion.
# These are case-insensitive substrings of the canonical domain_name strings
# defined in control_library_v0_2_descriptors_updated.json.
_DOMAIN_KEY_TRANSPARENCY = "transparency"   # D4: Transparency and Explainability
_DOMAIN_KEY_MONITORING   = "monitoring"     # D6: Monitoring and Incident Response


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ordered_union(*lists: list) -> list:
    seen: set = set()
    result: list = []
    for lst in lists:
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result


def _get_domain_score(
    domain_scores: dict,
    keyword: str,
    default: float = 5.0,
) -> float:
    """
    Find a domain score by case-insensitive keyword substring match.

    Returns ``default`` (5.0) when no matching domain is found, ensuring
    that missing domains never trigger unintended priority-promotion.
    """
    kw = keyword.lower()
    for name, score in domain_scores.items():
        if kw in name.lower():
            return float(score)
    return default


def _base_plan(risk_tier: str, is_llm: bool) -> list[str]:
    if is_llm:
        return {
            "minimal": ["llm_explainability"],
            "limited": ["llm_explainability", "llm_fairness"],
            "high":    ["llm_explainability", "llm_fairness",
                        "llm_uncertainty", "llm_drift"],
        }.get(risk_tier, [])
    return {
        "minimal": ["shap"],
        "limited": ["shap", "fairness"],
        "high":    ["shap", "fairness", "uncertainty", "drift"],
    }.get(risk_tier, [])


def _normalize_task_type(task_type: str) -> str:
    return str(task_type).strip().lower()


def _apply_task_compatibility_assessment(
    methods: list[str],
    task_type: str,
) -> tuple[list[str], dict]:
    task_type_normalized = _normalize_task_type(task_type)
    compatible_methods: list[str] = []
    incompatible_methods: list[str] = []

    for method in methods:
        allowed_tasks = METHOD_TASK_COMPATIBILITY.get(method)
        if allowed_tasks is None or task_type_normalized in allowed_tasks:
            compatible_methods.append(method)
        else:
            incompatible_methods.append(method)

    return compatible_methods, {
        "status": "applied",
        "task_type": task_type_normalized,
        "assessment_policy": "task-method compatibility screening",
        "compatible_methods": compatible_methods,
        "incompatible_methods": incompatible_methods,
        "compatible_count": len(compatible_methods),
        "incompatible_count": len(incompatible_methods),
        "compatibility_map_version": "v1",
        "note": (
            "Methods incompatible with the current task type were excluded "
            "after CBEP planning."
        ),
    }


# ---------------------------------------------------------------------------
# Primary public API  -- DTO-based
# ---------------------------------------------------------------------------

def plan_evidence(
    audit,
    governance=None,
) -> tuple[list[str], dict]:
    """
    Select the minimum sufficient evidence method set.

    Parameters
    ----------
    audit      : AuditContext DTO from S5 / AuditAdapter.
    governance : GovernanceContext DTO from S4 / GovernanceAdapter.
                 Optional; when absent no governance-driven adjustments apply.

    Returns
    -------
    (methods, trace)
    """
    is_llm = audit.system_type == "llm"
    task_type = _normalize_task_type(getattr(audit, "task_type", ""))

    # Step 1: base plan
    base = _base_plan(audit.risk_tier, is_llm)

    # Step 2: article-driven methods
    article_map = _S5_ARTICLE_LLM if is_llm else _S5_ARTICLE_TRAD
    article_methods: list[str] = []
    articles_matched: list[str] = []

    for article in audit.applicable_articles:
        mapped = article_map.get(article, [])
        if mapped:
            articles_matched.append(article)
            article_methods.extend(mapped)

    methods = _ordered_union(base, article_methods)

    # Step 3: governance-driven adjustments
    gov_adjustments: list[str] = []

    if governance is None:
        gov_trace: dict = {"status": "not_provided"}
    else:
        if governance.governance_score < _LOW_GOVERNANCE_THRESHOLD:
            sweep = _ALL_LLM if is_llm else _ALL_TRAD
            methods = _ordered_union(methods, sweep)
            gov_adjustments.append(
                f"governance_score={governance.governance_score:.2f} < "
                f"{_LOW_GOVERNANCE_THRESHOLD} -> full evidence sweep"
            )

        transparency = _get_domain_score(
            governance.domain_scores, _DOMAIN_KEY_TRANSPARENCY
        )
        if transparency < _PRIORITY_THRESHOLD:
            expl = _EXPLAINABILITY_LLM if is_llm else _EXPLAINABILITY_TRAD
            methods = _ordered_union(expl, methods)
            gov_adjustments.append(
                f"{_DOMAIN_KEY_TRANSPARENCY}_score={transparency:.2f} < "
                f"{_PRIORITY_THRESHOLD} -> explainability promoted"
            )

        monitoring = _get_domain_score(
            governance.domain_scores, _DOMAIN_KEY_MONITORING
        )
        if monitoring < _PRIORITY_THRESHOLD:
            drift = _DRIFT_LLM if is_llm else _DRIFT_TRAD
            methods = _ordered_union(drift, methods)
            gov_adjustments.append(
                f"{_DOMAIN_KEY_MONITORING}_score={monitoring:.2f} < "
                f"{_PRIORITY_THRESHOLD} -> drift promoted"
            )

        gov_trace = {
            "status":             "applied",
            "governance_score":   governance.governance_score,
            "governance_verdict": governance.governance_verdict,
            "domain_scores":      governance.domain_scores,
            "adjustments":        gov_adjustments,
        }

    # Step 4: CSP-driven extended sweep
    csp_adjustments: list[str] = []
    if not audit.csp_satisfied:
        sweep = _ALL_LLM if is_llm else _ALL_TRAD
        methods = _ordered_union(methods, sweep)
        csp_adjustments.append(
            "csp_satisfied=False -> extended evidence sweep enabled"
        )

    # Step 5: task-compatibility filtering
    task_compatibility_trace: dict
    if is_llm:
        task_compatibility_trace = {
            "status": "not_applied",
            "reason": "llm_path_uses_llm_specific_methods",
            "task_type": task_type,
            "assessment_policy": "task-method compatibility screening",
            "compatible_methods": methods,
            "incompatible_methods": [],
            "compatible_count": len(methods),
            "incompatible_count": 0,
            "compatibility_map_version": "v1",
            "note": (
                "Compatibility assessment is bypassed for LLM-specific "
                "evidence methods."
            ),
        }
    else:
        methods, task_compatibility_trace = _apply_task_compatibility_assessment(
            methods,
            task_type,
        )

    trace = {
        "cbep_version":            "3.0",
        "system_type":             audit.system_type,
        "task_type":               task_type,
        "modality":                audit.modality,
        "risk_tier":               audit.risk_tier,
        "application_domain":      audit.application_domain,
        "applicable_articles":     audit.applicable_articles,
        "articles_matched":        articles_matched,
        "blocking_findings_count": len(audit.blocking_findings),
        "csp_satisfied":           audit.csp_satisfied,
        "csp_adjustments":         csp_adjustments,
        "base_plan":               base,
        "article_plan":            list(dict.fromkeys(article_methods)),
        "governance_context":      gov_trace,
        "task_compatibility_assessment": task_compatibility_trace,
        "final_plan":              methods,
    }

    return methods, trace


# ---------------------------------------------------------------------------
# Backward-compatible wrappers  -- accept raw dicts, create DTOs internally
# ---------------------------------------------------------------------------

def select_methods_with_trace(
    risk_level:  str,
    system_type: str       = "traditional",
    s4_json:     dict | None = None,
    s5_json:     dict | None = None,
) -> tuple[list[str], dict]:
    """Backward-compatible wrapper around plan_evidence."""
    from adapters.s5_audit_adapter import AuditAdapter, AuditContext
    from adapters.s4_governance_adapter import GovernanceAdapter

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

    governance = (
        GovernanceAdapter.from_cgsa_report(s4_json) if s4_json else None
    )

    return plan_evidence(audit, governance)


def select_methods(
    risk_level:  str,
    system_type: str       = "traditional",
    s4_json:     dict | None = None,
    s5_json:     dict | None = None,
) -> list[str]:
    """Return only the method list.  Backward-compatible wrapper."""
    methods, _ = select_methods_with_trace(risk_level, system_type, s4_json, s5_json)
    return methods
